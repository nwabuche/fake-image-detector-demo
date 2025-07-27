TITLE:Fake image detection using CNN and Vit Transformer.

This project presents a deep learning-based pipeline to detect fake (GAN-generated) and real human face images.
 It compares.
            Project Highlight.
 Detects fake (GAN) vs real (CelebA) face images
 Compares CNN and ViT models
 Shows prediction result with confidence
 Provides interpretability via Grad-CAM 
 Web demo using  Streamlit

             Dataset
 The dataset includes:
 celeba_real_faces`: Real face images (CelebA)
 gan_fake_faces`: Fake face images (GAN-generated)
 A balanced dataset version with 500 real and 500 fake samples is also included for performance comparison.

Model used

CNN Based on ResNet18
ViT: Vision Transformer (ViT Small, patch size 16, embed_dim=384)

Streamlit Demo
Run the app locally:`bash(gitBash)
Upload any face image and get instant predictions from both models.