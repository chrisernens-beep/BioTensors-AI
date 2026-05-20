import numpy as np

from brain_architecture import BioTensor, ExecutiveCortex, BrocaArea


class BioBrainPro:
    """Multimodal SNN for health, navigation, and speech inputs."""

    def __init__(self, tokenizer=None):
        self.cortex_health = BioTensor((200, 100), name="Health_Primary")
        self.cortex_nav = BioTensor((200, 100), name="Nav_Primary")
        self.cortex_speech = BioTensor((200, 100), name="Speech_Primary")

        self.associative = BioTensor((400, 600), name="Associative_Global", threshold=0.8)

        self.executive = ExecutiveCortex(
            (100, 400),
            actions=["STABLE", "MANOEUVRE", "REPARER", "ALERTE"],
        )
        self.broca = BrocaArea((100, 400), name="Broca")
        self.limbic = BioTensor((100, 400), name="Limbic", threshold=1.2)

        self.arousal = 0.5
        self.internal_feedback = np.zeros(400, dtype=np.float32)
        self.speech_buffer = []
        self.tokenizer = tokenizer
        self.last_visual_input = None

    def forward_complex(self, health_in, nav_in, speech_in):
        """Process multimodal inputs in parallel."""
        h_out = self.cortex_health.forward(health_in)
        n_out = self.cortex_nav.forward(nav_in)
        s_out = self.cortex_speech.forward(speech_in)

        combined_spikes = np.concatenate([h_out, n_out, s_out])

        a_in = np.clip(combined_spikes + (self.internal_feedback * 0.2), 0, 1)
        a_out = self.associative.forward(a_in)
        self.internal_feedback = a_out

        self.executive.forward(a_out)
        self.broca.forward(a_out)

        return self.executive.get_decision()
