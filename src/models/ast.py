from transformers import ASTConfig, ASTForAudioClassifiction

def get_ast(num_classes=2):
    config = ASTConfig(num_labels=num_classes, num_mel_bins=128)
    model = ASTForAudioClassifiction(config)
    return model