from .DefenceContainer import DefenceContainer

class AppDomainContainer(DefenceContainer):
    def __init__(self, model_container, k1=9, k2=12, confidence=0.8):
        super(AppDomainContainer, self).__init__(model_container)

        params_received = {
            'k1': k1,
            'k2': k2,
            'confidence': confidence}
        self.defence_params.update(params_received)
    
    def fit(self, **kwargs):
        pass

    def defence(self, adv, **kwargs):
        pass