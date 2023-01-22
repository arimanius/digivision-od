import os
from typing import List, Set, Optional

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

from fairseq import tasks, options, checkpoint_utils, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from od.ofa.tasks.mm_tasks.refcoco import RefcocoTask
from od.util import getLogger

logger = getLogger('clip')

OFA_URLS = {
    "base": "https://huggingface.co/OFA-Sys/ofa-base-refcoco-fairseq-version/resolve/main/refcoco_base_best.pt"
}

OFA_SHA256S = {
    "base": "0d11081a07034745d55afc0ee3ffbdb9988ff0c2e852dd05560bf00e7398a5bd"
}

OFA_PATCH_SIZES = {
    "base": 384
}

MODELS_DIR = "~/.cache/ofa"


class OFAVisualGrounding:

    @staticmethod
    def download(model: str) -> str:
        if model not in OFA_URLS:
            raise ValueError(f"Invalid model {model}, must be one of {list(OFA_URLS.keys())}")
        url = OFA_URLS[model]
        model_dir = os.path.expanduser(MODELS_DIR)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{model}.pt')

        if not os.path.exists(model_path):
            logger.info(f"Downloading model {model} from {url} to {model_path}")
            torch.hub.download_url_to_file(url, model_path)
        return model_path

    def __get_symbols_to_strip_from_output(self) -> Set[int]:
        if hasattr(self.__generator, "symbols_to_strip_from_output"):
            return self.__generator.symbols_to_strip_from_output
        else:
            return {self.__generator.bos, self.__generator.eos}

    def __decode_fn(self, x: torch.Tensor) -> str:
        x = self.__task.tgt_dict.string(x.int().cpu(),
                                        extra_symbols_to_ignore=self.__get_symbols_to_strip_from_output())
        bin_result = []
        for token in x.strip().split():
            if token.startswith('<bin_'):
                bin_result.append(token)

        return ' '.join(bin_result)

    def __bin2coord(self, bins: str, w_resize_ratio: float, h_resize_ratio: float) -> List[float]:
        bin_list = [int(b[5:-1]) for b in bins.strip().split()]
        coord_list: List[float] = []
        coord_list += [bin_list[0] / (self.__task.cfg.num_bins - 1) * self.__task.cfg.max_image_size / w_resize_ratio]
        coord_list += [bin_list[1] / (self.__task.cfg.num_bins - 1) * self.__task.cfg.max_image_size / h_resize_ratio]
        coord_list += [bin_list[2] / (self.__task.cfg.num_bins - 1) * self.__task.cfg.max_image_size / w_resize_ratio]
        coord_list += [bin_list[3] / (self.__task.cfg.num_bins - 1) * self.__task.cfg.max_image_size / h_resize_ratio]
        return coord_list

    def __encode_text(self, text: str, length: Optional[int] = None, append_bos: bool = False,
                      append_eos: bool = False) -> torch.LongTensor:
        line = [
            self.__task.bpe.encode(' {}'.format(word.strip()))
            if not word.startswith('<code_') and not word.startswith('<bin_') else word
            for word in text.strip().split()
        ]
        line = ' '.join(line)
        s = self.__task.tgt_dict.encode_line(
            line=line,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.__bos_item, s])
        if append_eos:
            s = torch.cat([s, self.__eos_item])
        return s

    def __construct_sample(self, image: Image, instruction: str) -> dict:
        patch_image = self.__patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])

        instruction = self.__encode_text(' {}'.format(instruction.lower().strip()), append_bos=True,
                                         append_eos=True).unsqueeze(0)
        instruction_length = torch.LongTensor([s.ne(self.__pad_idx).long().sum() for s in instruction])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": instruction,
                "src_lengths": instruction_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            }
        }
        return sample

    @staticmethod
    def __apply_half(t: torch.Tensor) -> torch.Tensor:
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def __init__(self, model: str, cuda: bool, bpe_dir: str):
        model_path = self.download(model)
        tasks.register_task('refcoco', RefcocoTask)
        self.__use_fp16 = self.__use_cuda = cuda and torch.cuda.is_available()

        # specify some options for evaluation
        parser = options.get_generation_parser()
        input_args = ["", "--task=refcoco", "--beam=10", f"--path={model_path}",
                      f"--bpe-dir={bpe_dir}", "--no-repeat-ngram-size=3",
                      f"--patch-image-size={OFA_PATCH_SIZES[model]}"]
        args = options.parse_args_and_arch(parser, input_args)
        cfg = convert_namespace_to_omegaconf(args)

        # Load pretrained ckpt & config
        self.__task = tasks.setup_task(cfg.task)
        self.__models, cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            task=self.__task,
        )

        # Move models to GPU
        for model in self.__models:
            model.eval()
            if self.__use_fp16:
                model.half()
            if self.__use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        # Initialize generator
        self.__generator = self.__task.build_generator(self.__models, cfg.generation)

        # Image transform
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.__patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.__task.cfg.patch_image_size, self.__task.cfg.patch_image_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Text preprocess
        self.__bos_item = torch.LongTensor([self.__task.src_dict.bos()])
        self.__eos_item = torch.LongTensor([self.__task.src_dict.eos()])
        self.__pad_idx = self.__task.src_dict.pad()

    def __call__(self, image: Image.Image, instruction: str) -> List[float]:
        w, h = image.size
        instruction = f'which region does the text " {instruction} " describe?'

        sample = self.__construct_sample(image, instruction)
        sample = utils.move_to_cuda(sample) if self.__use_cuda else sample
        sample = utils.apply_to_sample(self.__apply_half, sample) if self.__use_fp16 else sample

        hypos = self.__task.inference_step(self.__generator, self.__models, sample)
        bins = self.__decode_fn(hypos[0][0]["tokens"])

        w_resize_ratio = self.__task.cfg.patch_image_size / w
        h_resize_ratio = self.__task.cfg.patch_image_size / h

        coord_list = self.__bin2coord(bins, w_resize_ratio, h_resize_ratio)
        return coord_list


class OFADetector:

    @classmethod
    def load(cls, model: str) -> None:
        OFAVisualGrounding.download(model)

    def __init__(self, model: str, cuda: bool, bpe_dir: str, instruction: str):
        self.__model = OFAVisualGrounding(model, cuda, bpe_dir)
        self.__instruction = instruction

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    @torch.inference_mode()
    def detect(self, image: Image.Image) -> List[float]:
        return self.__model(image, self.__instruction)
