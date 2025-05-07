import torch
import argparse
import os
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run zero-shot inference with CT-CLIP model')
    
    parser.add_argument('--checkpoint', type=str, 
                        default="/mnt/home/admvkl@median.cad/code/public/CT-CLIP/scripts/output_folder/CTClip.28000.pt",
                        help='Path to the model checkpoint')
    
    parser.add_argument('--data-folder', type=str,
                        default='/mnt/home/admvkl@median.cad/code/public/example_download_script/data_volumes/dataset/valid_preprocessed/',
                        help='Path to the data folder')
    
    parser.add_argument('--reports-file', type=str,
                        default="/mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
                        help='Path to the reports file')
    
    parser.add_argument('--labels', type=str,
                        default="/mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
                        help='Path to the labels file')
    
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for inference')
    
    parser.add_argument('--results-folder', type=str, default="inference_zeroshot_28000/",
                        help='Where to save results')
    
    parser.add_argument('--num-train-steps', type=int, default=1,
                        help='Number of training steps')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--wd', type=float, default=0.0,
                        help='Weight decay')
    
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Maximum gradient norm')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Print the parameters being used
    print("Running zero-shot inference with the following parameters:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data folder: {args.data_folder}")
    print(f"  Reports file: {args.reports_file}")
    print(f"  Labels file: {args.labels}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Results folder: {args.results_folder}")
    print(f"  Number of training steps: {args.num_train_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.wd}")
    print(f"  Max gradient norm: {args.max_grad_norm}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_folder, exist_ok=True)
    
    # Initialize tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize image encoder
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

    # Initialize CLIP model
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

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Remove `module.` prefix if it exists
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")  # Remove 'module.' from key names
        new_state_dict[new_key] = v

    clip.load_state_dict(new_state_dict, strict=False)  # strict=False to avoid issues

    # Initialize inference
    print(f"Initializing inference...")
    inference = CTClipInference(
        clip,
        data_folder = args.data_folder,
        reports_file = args.reports_file,
        labels = args.labels,
        batch_size = args.batch_size,
        results_folder = args.results_folder,
        num_train_steps = args.num_train_steps,
        lr = args.lr,
        wd = args.wd,
        max_grad_norm = args.max_grad_norm,
    )

    # Run inference
    print(f"Running inference...")
    inference.infer()
    print(f"Inference completed. Results saved to {args.results_folder}")

if __name__ == "__main__":
    main()
