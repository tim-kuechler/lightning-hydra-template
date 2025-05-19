from typing import Any, Dict, Optional

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only


class BaseImageLogger(Callback):
    """A basic image logger callback. Samples images and logs them to disc and wandb."""

    def __init__(
            self,
            num_batches: int = 1,
            sampling_batch_size: int = 1,
            split: str = "val",
            shuffle: bool = True,
            sampling_freq: int = 1000,
        ) -> None:
        """Init a BaseImageLogger

        :param num_batches: The number of batches to sample.
        :param sampling_batch_size: The batch size to be used for sampling.
        :param split: The dataloader to use for sampling. One of ("train", "val", "test")
        :param shuffle: If true, the dataloader is shuffled.
        :param sampling_freq: Sample every n-th training step.
        :param cond: Sample images conditionally (i.e. with prompt and without CFG).
        :param uncond: Sample images unconditionally (i.e. without prompt / with "" prompt).
        :param cfg: Sample images with Classifier-free guidance (with prompts).
        """
        super().__init__()

        self.num_batches = num_batches
        self.sampling_batch_size = sampling_batch_size
        self.split = split
        self.shuffle = shuffle
        self.sampling_freq = sampling_freq

        self.last_step = 0

    @rank_zero_only
    def on_train_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, _, __: Dict[str, Any], ___: int
    ) -> None:
        """If th specified sampling frequency is reached, sample images according to specifications and save them
        to disc and to wandb if initialized.

        :param trainer: PyTorch Lightning Trainer.
        :param pl_module: PyTorch Lightning Module.
        """
        if trainer.global_step % self.sampling_freq == 0 and self.last_step != trainer.global_step:
            # sample_images(
            #     pl_module,
            #     trainer,
            #     save_wandb=True,
            #     save_local=False,
            #     num_batches=self.num_batches,
            #     batch_size=self.sampling_batch_size,
            #     modes=modes,
            #     split=self.split,
            #     shuffle=self.shuffle,
            # )

            self.last_step = trainer.global_step


class BaseTestCallback(Callback):
    """A basic logger callback that runs after training finishes. Samples the same number of images as
    in the test set and calculates FID, KID and CMMD."""

    def __init__(
            self,
            eval_cmmd: bool = True,
            eval_fid: bool = True,
            eval_kid: bool = True,
            cond: bool = False,
            uncond: bool = False,
            cfg: bool = False,
            metric_batch_size: int = 32,
            sampling_batch_size: Optional[int] = None,
    ):
        """Init a BaseTestCallback

        :param eval_cmmd: If true, evaluate CMMD.
        :param eval_fid: If true, evaluate FID.
        :param eval_kid: If true, evaluate KID.
        :param cond: Sample images conditionally (i.e. with prompt and without CFG).
        :param uncond: Sample images unconditionally (i.e. without prompt / with "" prompt).
        :param cfg: Sample images with Classifier-free guidance (with prompts).
        :param metric_batch_size: The batch size used during metric calculation.
        :param sampling_batch_size: The batch size used during test image sampling.
        """
        self.eval_cmmd = eval_cmmd
        self.eval_fid = eval_fid
        self.eval_kid = eval_kid
        self.cond = cond
        self.uncond = uncond
        self.cfg = cfg
        self.metric_batch_size = metric_batch_size
        self.sampling_batch_size = sampling_batch_size

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Callback that activate when training ends and runs the test evaluation.

        :param trainer: PyTorch Lightning Trainer.
        :param pl_module: PyTorch Lightning Module.
        """
        # calculate_metrics(
        #     pl_module,
        #     trainer,
        #     self.eval_fid,
        #     self.eval_kid,
        #     self.eval_cmmd,
        #     "test",
        #     None,
        #     self.cond,
        #     self.uncond,
        #     self.cfg,
        #     False,
        #     None,
        #     self.metric_batch_size,
        #     True,
        #     self.sampling_batch_size,
        # )


class BaseEvalCallback(Callback):
    """A basic logger callback that runs during trainig. Samples a small number of validation images
     and calculates FID, KID and CMMD."""

    def __init__(
            self,
            eval_freq: int = 1000,
            eval_cmmd: bool = True,
            eval_fid: bool = False,
            eval_kid: bool = False,
            cond: bool = False,
            uncond: bool = False,
            cfg: bool = False,
            metric_batch_size: int = 32,
            num_images: int = 100,
            delete_samples: bool = True,
            sampling_batch_size: Optional[int] = None,
    ):
        """Init a BaseEvalCallback

        :param eval_freq: Run the evaluation every n-th training step.
        :param eval_cmmd: If true, evaluate CMMD.
        :param eval_fid: If true, evaluate FID.
        :param eval_kid: If true, evaluate KID.
        :param cond: Sample images conditionally (i.e. with prompt and without CFG).
        :param uncond: Sample images unconditionally (i.e. without prompt / with "" prompt).
        :param cfg: Sample images with Classifier-free guidance (with prompts).
        :param metric_batch_size: The batch size used during metric calculation.
        :param num_images: The number of validation images to sample.
        :param delete_samples: If True, delete the generated samples after score computation.
        :param sampling_batch_size: The batch size used during test image sampling.
        """
        self.eval_freq = eval_freq
        self.eval_cmmd = eval_cmmd
        self.eval_fid = eval_fid
        self.eval_kid = eval_kid
        self.cond = cond
        self.uncond = uncond
        self.cfg = cfg
        self.metric_batch_size = metric_batch_size
        self.num_images = num_images
        self.delete_samples = delete_samples
        self.sampling_batch_size = sampling_batch_size

        self.last_step = 0

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, _, __: Dict[str, Any], ___: int
    ) -> None:
        """Callback that runs every n-th step as specified and calculates FID, KID and CMMD on generated validation
        images.

        :param trainer: PyTorch Lightning Trainer.
        :param pl_module: PyTorch Lightning Module.
        """
        if trainer.global_step % self.eval_freq == 0 and self.last_step != trainer.global_step:
            # calculate_metrics(
            #     pl_module,
            #     trainer,
            #     self.eval_fid,
            #     self.eval_kid,
            #     self.eval_cmmd,
            #     "val",
            #     trainer.global_step,
            #     self.cond,
            #     self.uncond,
            #     self.cfg,
            #     self.delete_samples,
            #     self.num_images,
            #     self.metric_batch_size,
            #     True,
            #     self.sampling_batch_size,
            # )

            self.last_step = trainer.global_step
