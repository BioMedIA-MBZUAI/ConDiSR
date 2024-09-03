# ConDiSR
Official Repository for the paper titled ConDiSR: Contrastive Disentanglement and Style Regularization for Single Domain Generalization. [Link to the paper](https://arxiv.org/html/2403.09400v1)

# Abstract:
Medical data often exhibits distribution shifts, leading to performance degradation of deep learning models trained using standard supervised learning pipelines. Domain Generalization (DG) addresses this challenge, with Single-Domain Generalization (SDG) being notably relevant due to the privacy and logistical constraints often inherent in medical data. Existing disentanglement-based SDG methods heavily rely on structural information from segmentation masks, but classification labels do not offer similarly dense information. This work introduces a novel SDG method for medical image classification, utilizing channel-wise contrastive disentanglement. The method is further refined with reconstruction-based style regularization to ensure distinct style and structural feature representations are extracted. We evaluate our method on the complex tasks of multicenter histopathology image classification and Diabetic Retinopathy (DR) grading in fundus images, benchmarking it against state-of-the-art (SOTA) SDG baselines. Our results demonstrate that our method consistently outperforms the SOTA independently on the choice of the source domain while exhibiting greater performance stability. This study underscores the importance and challenges of exploring SDG frameworks for classification tasks.

<p align="center">
    <img src="./figures/wacv_model_fig_v3.png" width="78%" />
</p>

The application code we use is based on backbone codes from DomainBed[1], MIRO[2] and SWAD[3].

Install required libraries:
```
pip install -r requirements.txt
```

Run training with Camelyon17_WILDS datasets (have to be organized similarly to PACS):
```
python tools/train.py --root /dataset/path --source_domains s --target_domains t1 t2 t3 t4 --dataset-config-file ./configs/datasets/dg/cam17.yaml --config-file ./configs/trainers/dg/vanilla/adgv_cam17.yaml
```
