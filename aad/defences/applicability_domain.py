import logging
import time

from .DefenceContainer import DefenceContainer

logger = logging.getLogger(__name__)


class ApplicabilityDomainContainer(DefenceContainer):
    def __init__(self, model_container, k1=9, k2=12, confidence=0.8):
        super(ApplicabilityDomainContainer, self).__init__(model_container)

        params_received = {
            'k1': k1,
            'k2': k2,
            'confidence': confidence}
        self.defence_params.update(params_received)

    def fit(self, **kwargs):
        # TODO: implement Stage 1
        logger.warning('Stage 1 for Applicability Domain is NOT impletmented!')
        self._log_time_start()
        time.sleep(2)
        self._log_time_end('train AD')
        return True

    def defence(self, adv, **kwargs):
        pass
