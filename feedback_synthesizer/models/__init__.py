from .tacotron import Tacotron



def create_model(name, hparams, resnet_scope, resnet_hp):
    if name == "Tacotron":
        return Tacotron(hparams, resnet_scope, resnet_hp)
    else:
        raise Exception("Unknown model: " + name)
'''
def create_model(name, hparams, embed_net):
    if name == "Tacotron":
        return Tacotron(hparams, embed_net)
    else:
        raise Exception("Unknown model: " + name)
'''