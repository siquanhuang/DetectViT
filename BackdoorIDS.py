import random
from PIL import Image, ImageFilter
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
from torchvision import transforms

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_vit = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

class BackdoorIDS():
    def __init__(self, model, processor=None, device=None, sacle_k = 20, neighbour = 4, grid = 16, remove_mode = "mean"):
        self.model = model
        self.processor = processor
        self.grid = grid
        self.remove_mode = remove_mode
        self.sacle_k = sacle_k
        self.neighbour = neighbour
        self.device = device

    def compute_embedding_img_list(self, model, proc=None, img_list=[]):
        model.eval()
        with torch.no_grad():
            if proc is not None:
                if hasattr(proc, "process_image"):
                    t = torch.stack([proc.process_image(img_pil).to(self.device) for img_pil in img_list]).to(self.device)
                else:
                    t = proc(images=img_list, return_tensors='pt')['pixel_values'].to(self.device)
                feat = model.get_image_features(t)
            else:
                if hasattr(model, 'forward_features'):
                    img_tensor_list = [test_transform_vit(img) for img in img_list]
                    t = torch.stack(img_tensor_list).to(self.device)
                    # DeiT / ViT：用 forward_features 取 cls token embedding
                    out = model.forward_features(t)
                    feat = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    img_tensor_list = [test_transform(img) for img in img_list]
                    t = torch.stack(img_tensor_list).to(self.device)
                    feat, out = model(t)
            return feat

    def cosine_sim(self, a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-12) * (np.linalg.norm(b)+1e-12)))

    def mask_patches(self, img_pil, patch_indices, grid, mode="mean"):
        img = img_pil.copy()
        W, H = img.size
        patch_w = W // grid
        patch_h = H // grid
        mean_color = tuple(map(int, np.array(img).mean(axis=(0,1)).tolist()))
        for (r, c) in patch_indices:
            left = c * patch_w
            upper = r * patch_h
            right = (c+1) * patch_w if c < grid-1 else W
            lower = (r+1) * patch_h if r < grid-1 else H
            box = (left, upper, right, lower)
            if mode == "mean":
                block = Image.new("RGB", (right-left, lower-upper), color=mean_color)
                img.paste(block, box)
            elif mode == "black":
                block = Image.new("RGB", (right-left, lower-upper), color=(0,0,0))
                img.paste(block, box)
            elif mode == "blur":
                patch = img.crop(box).filter(ImageFilter.GaussianBlur(radius=7))
                img.paste(patch, box)
            else:
                raise ValueError("未知 mode")
        return img

    def project_embeddings_pca(self, embeddings, steps, n_components=2):
        embeddings = np.array([emb.flatten() for emb in embeddings])
        cosine_dist_matrix = cosine_distances(embeddings)
        print(embeddings.shape)

        tsne = TSNE(n_components=n_components, metric='cosine', random_state=42)
        pca_result = tsne.fit_transform(embeddings)

        colormap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=0, vmax=max(steps))

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=steps, cmap=colormap, norm=norm, marker='o')
        cbar = plt.colorbar(scatter)

        cbar.set_label('Masking Step', fontsize=25)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("test.pdf",dpi=300)
        plt.show()

        return pca_result


    def calculate_knn_density(self, embeddings):
        cosine_dist_matrix = cosine_distances(embeddings)
        knn = NearestNeighbors(n_neighbors=self.neighbour, metric='precomputed')
        knn.fit(cosine_dist_matrix)
        distances, _ = knn.kneighbors(cosine_dist_matrix)

        density = np.mean(distances, axis=1)
        return density


    def cluster_varients(self, embeddings):
        density = self.calculate_knn_density(embeddings[:self.neighbour+1])

        p = density.mean()
        eps = self.sacle_k * p
        if eps < 0.000000001:
            eps = 0.000000001

        dbscan = DBSCAN(eps=eps, min_samples=self.neighbour, metric='precomputed')
        cosine_dist_matrix = cosine_distances(embeddings)
        labels = dbscan.fit_predict(cosine_dist_matrix)

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            return True
        else:
            return False

        return backdoor_samples, labels

    def get_image_varients(self, orig_img, all_patches, steps_to_run):
        # random order once
        patch_order = all_patches.copy()
        random.shuffle(patch_order)

        current_img = orig_img.copy()

        images_list = [current_img]
        for step in range(int(0.75 * steps_to_run - 1)):
            next_patch = patch_order[step]
            current_img = self.mask_patches(current_img, [next_patch], self.grid, mode=self.remove_mode)
            images_list.append(current_img)

        return images_list


    def detect_backdoor_samples(self, image_pil):
        max_steps = self.grid * self.grid
        orig_img = image_pil

        all_patches = [(r, c) for r in range(self.grid) for c in range(self.grid)]
        n_patches = len(all_patches)
        steps_to_run = min(max_steps, n_patches)

        images_list = self.get_image_varients(orig_img, all_patches, steps_to_run)

        emb_list = self.compute_embedding_img_list(self.model, self.processor, images_list)

        flag = self.cluster_varients(emb_list.cpu().numpy())

        return flag

    def detect_backdoor_samples_dataset(self, dataset):
        results = {}
        self.model.eval()
        with torch.no_grad():
            for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"): # BadViT, TrojViT, BadCLIP
            # for idx, image in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):  #BadVision
                image, _ = dataset[idx]  # BadViT, TrojViT, BadCLIP
                # image = dataset[idx]  # BadVision

                image = image.numpy().transpose(1, 2, 0)

                if image.dtype == 'float32':
                    image = (image * 255).clip(0, 255).astype('uint8')

                image_pil = Image.fromarray(image)
                result = self.detect_backdoor_samples(image_pil)
                results[idx] = result
                # print(f"Processing sample {idx + 1} of {len(dataset)}")
        return results


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


