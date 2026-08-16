import segmentation_models_pytorch as smp

ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def get_model(architecture, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Architettura sconosciuta: {architecture}. Disponibili: {list(ARCHITECTURES)}")

    model_cls = ARCHITECTURES[architecture]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # restiamo in logit space, la sigmoid e' gestita da loss/metriche
    )
