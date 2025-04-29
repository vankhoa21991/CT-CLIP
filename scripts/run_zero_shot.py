import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)



# clip.load("/mnt/home/admvkl@median.cad/code/public/CT-RATE/models/CT-CLIP-Related/CT-CLIP_v2.pt")

import torch

# Load checkpoint
checkpoint = torch.load("/mnt/home/admvkl@median.cad/code/public/CT-CLIP/scripts/output_folder/CTClip.25000.pt", map_location="cpu")

# Remove `module.` prefix if it exists
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace("module.", "")  # Remove 'module.' from key names
    new_state_dict[new_key] = v

clip.load_state_dict(new_state_dict, strict=False)  # strict=False to avoid issues

inference = CTClipInference(
    clip,
    data_folder = '/mnt/home/admvkl@median.cad/code/public/example_download_script/data_volumes/dataset/valid_preprocessed/',
    reports_file= "/mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
    labels = "/mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    batch_size = 4,
    results_folder="inference_zeroshot_new/",
    num_train_steps = 1,
)

inference.infer()
