from gym.envs.registration import register

register(
    id='speech-resynthesis-mfcc-v0',
    entry_point='gym_speech_resynthesis.envs:SpeechResynthesisEnvMFCC',
)

