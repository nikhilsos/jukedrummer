HPARAMS_REGISTRY = {}
MODEL_LIST = {
    1: ("vq1", "lm1"),
    2: ("vq1", "lm2"),
    3: ("vq1", "lm3"),
    4: ("vq1", "lm4"),
    5: ("vq1", "lm5"),
    6: ("vq1", "lm6"),
    7: ("vq1", "lm7"),
    8: ("vq1", "lm8"),
    9: ("vq2", "lm1"),
    10: ("vq3", "lm1"),
    11: ("vq1", "lm9"),
    12: ("vq1", "lm10"),
    21: ("vq4", "lm1"),
    22: ("vq4", "lm9"),
    23: ("vq1", "lm11"),
    24: ("vq1", "lm12"),
    25: ("vq5", "lm12"),
    26: ("vq6", "lm12"),
    27: ("vq7", "lm13"),
    28: ("vq8", "lm14"),
    29: ("vq7", "lm15"),
    30: ("vq7", "lm16"),
    31: ("vq7", "lm17"),
    32: ("vq8", "lm18"),
}


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value


LM_DEFAULTS = Hyperparams(
    batch_size=4,
    sample_step=10,
    ckpt_dir="ckpt/",
    path="data_jd/",
)

VQ_DEFAULTS = Hyperparams(
    batch_size=4,
    sample_step=50,
    ckpt_dir="ckpt/",
    path="data/",
)


def setup_lm_hparams(hparam_set_names):
    # hparam_set_names = (vqvae_hparam, lm_hparam)
    H = Hyperparams()
    vqvae_sets = HPARAMS_REGISTRY[hparam_set_names[0].strip()]
    lm_sets = HPARAMS_REGISTRY[hparam_set_names[1].strip()]
    H.update(LM_DEFAULTS)
    H.update(vqvae_sets)
    H["vq_name"] = H["name"]
    H.update(lm_sets)

    return H


def setup_vq_hparams(hparam_set_name):
    # hparam_set_name = vqvae_hparam
    print(f"set up {hparam_set_name} hparam")
    H = Hyperparams()
    vqvae_sets = HPARAMS_REGISTRY[hparam_set_name.strip()]
    H.update(VQ_DEFAULTS)
    H.update(vqvae_sets)

    return H


OPT = Hyperparams(
    epochs=1000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.05,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
    fp16_opt=False,
    prior=True,
    restore_prior="",
    fp16=False,
)

MEL = Hyperparams(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    sampling_rate=44100,
    n_mel_channels=80,
    extension=".wav",
    mel_fmin=0.0,
    mel_fmax=None,
)

vq1 = Hyperparams(
    # codebook 512 | 1024
    name="vq1",
    codebook_size=512,
    upsample_ratios=[2, 2],
    downsample_ratios=[0.5, 0.5],
    commit_beta=0.25,
)
HPARAMS_REGISTRY["vq1"] = vq1

vq2 = Hyperparams(
    # codebook 64 | 512
    name="vq2",
    codebook_size=64,
    upsample_ratios=[4, 2],
    downsample_ratios=[0.5, 0.25],
    commit_beta=0.25,
)
HPARAMS_REGISTRY["vq2"] = vq2

vq3 = Hyperparams(
    # codebook 32 | 2048
    name="vq3",
    batch_size=6,
    codebook_size=32,
    upsample_ratios=[2, 1],
    downsample_ratios=[1, 0.5],
    commit_beta=0.02,
)
HPARAMS_REGISTRY["vq3"] = vq3

vq4 = Hyperparams(
    # codebook 32 | 1024, indentical to vq1 but differnet dataset
    name="vq4",
    codebook_size=32,
    upsample_ratios=[2, 2],
    downsample_ratios=[0.5, 0.5],
    commit_beta=0.02,
)
HPARAMS_REGISTRY["vq4"] = vq4

vq5 = Hyperparams(
    # Full dataset: 4x, codebook 64 (target) / 256 (others)
    name="vq5",
    path="data_jd/",
    codebook_size=64,
    upsample_ratios=[2, 2],
    downsample_ratios=[0.5, 0.5],
    commit_beta=0.25,
)
HPARAMS_REGISTRY["vq5"] = vq5

vq6 = Hyperparams(
    # Full dataset: 4x, codebook 128, latent 128, transient loss
    name="vq6",
    path="data_jd/",
    codebook_size=128,
    latent_dim=128,
    upsample_ratios=[2, 2],
    downsample_ratios=[0.5, 0.5],
    commit_beta=0.25,
)
HPARAMS_REGISTRY["vq6"] = vq6

vq7 = Hyperparams(
    # Full dataset: 4x, codebook 32, latent 128, energy-weighted commitment
    name="vq7",
    path="data_jd/",
    codebook_size=32,
    latent_dim=128,
    upsample_ratios=[2, 2],
    downsample_ratios=[0.5, 0.5],
    commit_beta=0.25,
)
HPARAMS_REGISTRY["vq7"] = vq7

vq8 = Hyperparams(
    # Full dataset: 4x, codebook 128, latent 128, energy-weighted commitment
    name="vq8",
    path="data_jd/",
    codebook_size=128,
    latent_dim=128,
    upsample_ratios=[2, 2],
    downsample_ratios=[0.5, 0.5],
    commit_beta=0.25,
)
HPARAMS_REGISTRY["vq8"] = vq8

lm1 = Hyperparams(
    # Raw beat activation (Low-level) w/o in-attention
    name="lm1",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.4,
    heads=4,
    blocks=16,
    num_classes=4,
    binfo_type="mid",  # default = low
)
HPARAMS_REGISTRY["lm1"] = lm1

lm2 = Hyperparams(
    # No beat activation w/o in-attention
    name="lm2",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type=None,
)
HPARAMS_REGISTRY["lm2"] = lm2

lm3 = Hyperparams(
    # Raw beat activation (Low-level) w/ in-attention
    name="lm3",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="low",
)
HPARAMS_REGISTRY["lm3"] = lm3

lm4 = Hyperparams(
    # lm1 but d_model=1024
    name="lm4",
    batch_size=8,
    enc_layers=20,
    dec_layers=9,
    d_model=1024,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="low",
)
HPARAMS_REGISTRY["lm4"] = lm4

lm5 = Hyperparams(
    # beat / downbeat / non-beat 3 token beat embed (High-level) w/o in-attention
    name="lm5",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="high",
)
HPARAMS_REGISTRY["lm5"] = lm5

lm6 = Hyperparams(
    # beat / downbeat / non-beat 3 token beat embed (High-level) w/ in-attention
    name="lm6",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="high",
)
HPARAMS_REGISTRY["lm6"] = lm6

lm7 = Hyperparams(
    # onset/ non-onset 2 tokens beat embed (Mid-level) w/o in-attention
    name="lm7",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="mid",
)
HPARAMS_REGISTRY["lm7"] = lm7

lm8 = Hyperparams(
    # onset/ non-onset 2 tokens beat embed (Mid-level) w/ in-attention
    name="lm8",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="mid",
)
HPARAMS_REGISTRY["lm8"] = lm8

lm9 = Hyperparams(
    # No encoder w/ raw beat info (Low-level)
    name="lm9",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type="low",
)
HPARAMS_REGISTRY["lm9"] = lm9

lm10 = Hyperparams(
    # No encoder w/o beat info
    name="lm10",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.2,
    heads=2,
    blocks=16,
    binfo_type=None,
)
HPARAMS_REGISTRY["lm10"] = lm10

lm11 = Hyperparams(
    # Pansori downbeat phase (dphase) conditioning via sin/cos projection
    name="lm11",
    path="data/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.4,
    heads=4,
    blocks=16,
    num_classes=4,
    binfo_type="dphase",
    lambda_p=0.2,  # perceptual loss weight
    tau=0.5,  # softmax temperature for soft codebook lookup (lower = sharper)
    focal_gamma=2.0,  # focal loss focusing parameter (0 = standard CE)
    lambda_fad=0.01,  # FAD loss weight
    fad_diagonal=True,  # diagonal covariance (stable); set False for full covariance
    perceptual_type="advanced",  # 'standard', 'advanced', or 'multiscale'
    perceptual_alpha=0.5,  # weight for cosine component (only for 'advanced')
    perceptual_taus=[
        0.1,
        0.5,
        1.0,
    ],  # temperatures for multiscale (only for 'multiscale')
)
HPARAMS_REGISTRY["lm11"] = lm11

lm12 = Hyperparams(
    # Jungmori-only, no class conditioning
    name="lm12",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.4,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    lambda_p=0.2,
    tau=0.5,
    focal_gamma=2.0,
    lambda_fad=0.01,
    fad_diagonal=True,
    perceptual_type="advanced",
    perceptual_alpha=0.5,
    perceptual_taus=[0.1, 0.5, 1.0],
)
HPARAMS_REGISTRY["lm12"] = lm12

lm13 = Hyperparams(
    # Jungmori-only, no class conditioning, vq7 (codebook=32, energy-weighted commit)
    name="lm13",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.4,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    lambda_p=0.2,
    tau=0.5,
    focal_gamma=2.0,
    lambda_fad=0.01,
    fad_diagonal=True,
    perceptual_type="advanced",
    perceptual_alpha=0.5,
    perceptual_taus=[0.1, 0.5, 1.0],
    vq_ckpt_tag="target_energy",  # loads vq7_target_energy.pkl
)
HPARAMS_REGISTRY["lm13"] = lm13

lm14 = Hyperparams(
    # Jungmori-only, no class conditioning, vq8 (codebook=128, energy-weighted commit)
    name="lm14",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.4,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    lambda_p=0.2,
    tau=0.5,
    focal_gamma=2.0,
    lambda_fad=0.01,
    fad_diagonal=True,
    perceptual_type="advanced",
    perceptual_alpha=0.5,
    perceptual_taus=[0.1, 0.5, 1.0],
    vq_ckpt_tag="target_energy",  # loads vq8_target_energy.pkl
)
HPARAMS_REGISTRY["lm14"] = lm14

lm15 = Hyperparams(
    # Jungmori-only, vq7. Vocal (others) tokens injected per-timestep into x_cond
    # so the drums actually condition on the vocal; low dropout so the conditioning
    # path can learn. Train with plain CE (no --percep/--onset/--fad/--hit_boost).
    name="lm15",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.1,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    vocal_xcond=True,
    others_codebook_size=256,  # vq7 others stem uses a 256-entry codebook
    vq_ckpt_tag="target_energy",  # loads vq7_target_energy.pkl
)
HPARAMS_REGISTRY["lm15"] = lm15

lm16 = Hyperparams(
    # Like lm15 but vocal conditioning = continuous `others` mel into x_cond.
    name="lm16",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.1,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    vocal_feat="mel",
    vq_ckpt_tag="target_energy",
)
HPARAMS_REGISTRY["lm16"] = lm16

lm17 = Hyperparams(
    # Like lm15 but vocal conditioning = vocal energy + onset envelope into x_cond.
    name="lm17",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.1,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    vocal_feat="envelope",
    vq_ckpt_tag="target_energy",
)
HPARAMS_REGISTRY["lm17"] = lm17

lm18 = Hyperparams(
    # vq8 (codebook=128) + winning vocal feature (mel) into x_cond.
    name="lm18",
    path="data_jungmori/",
    enc_layers=20,
    dec_layers=9,
    d_model=512,
    dropout=0.1,
    heads=4,
    blocks=16,
    binfo_type="dphase",
    vocal_feat="mel",
    vq_ckpt_tag="target_energy",  # loads vq8_target_energy.pkl
)
HPARAMS_REGISTRY["lm18"] = lm18
