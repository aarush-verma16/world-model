"""One-off: drop superseded entries from refs.bib and append verified 2024-2026 entries."""

import re
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "latex" / "refs.bib"

DROP = {
    "kaiser2020simple", "okada2021dreaming", "seo2023mwm", "lample2017fps", "mirowski2017navigate",
    "dosovitskiy2017dfp", "gupta2017cmp", "parisotto2018neuralmap", "argall2009survey",
    "rajeswaran2018dapg", "brittain2019pser", "andrychowicz2017her", "ecoffet2021firstreturn",
    "chen2019lbc", "hafner2022director", "walker2023modelbased",
}

NEW = r"""
% ---------------------------------------------------------------- 2024-2026 additions (verified 2026-11-24)
@article{ding2025survey,
  title   = {Understanding World or Predicting Future? {A} Comprehensive Survey of World Models},
  author  = {Ding, Jingtao and Zhang, Yunke and Shang, Yu and Zhang, Yuheng and Zong, Zefang and Feng, Jie and Yuan, Yuan and Su, Hongyuan and Li, Nian and Sukiennik, Nicholas and Xu, Fengli and Li, Yong},
  journal = {ACM Computing Surveys},
  volume  = {58},
  year    = {2025},
  doi     = {10.1145/3746449}
}
@inproceedings{richens2025general,
  title     = {General agents need world models},
  author    = {Richens, Jonathan and Everitt, Tom and Abel, David},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {51659--51687},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/richens25a.html}
}
@inproceedings{samsami2024r2i,
  title     = {Mastering Memory Tasks with World Models},
  author    = {Samsami, Mohammad Reza and Zholus, Artem and Rajendran, Janarthanan and Chandar, Sarath},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  eprint    = {2403.04253},
  archivePrefix = {arXiv}
}
@inproceedings{burchi2025twister,
  title     = {Learning Transformer-based World Models with Contrastive Predictive Coding},
  author    = {Burchi, Maxime and Timofte, Radu},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  eprint    = {2503.04416},
  archivePrefix = {arXiv}
}
@misc{burchi2024mudreamer,
  title         = {{MuDreamer}: Learning Predictive World Models without Reconstruction},
  author        = {Burchi, Maxime and Timofte, Radu},
  year          = {2024},
  eprint        = {2405.15083},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2405.15083}
}
@inproceedings{valevski2025gamengen,
  title     = {Diffusion Models Are Real-Time Game Engines},
  author    = {Valevski, Dani and Leviathan, Yaniv and Arar, Moab and Fruchter, Shlomi},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  eprint    = {2408.14837},
  archivePrefix = {arXiv}
}
@inproceedings{bruce2024genie,
  title     = {Genie: Generative Interactive Environments},
  author    = {Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and others},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {4603--4623},
  publisher = {PMLR},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/bruce24a.html}
}
@inproceedings{zhou2025dinowm,
  title     = {{DINO-WM}: World Models on Pre-trained Visual Features enable Zero-shot Planning},
  author    = {Zhou, Gaoyue and Pan, Hengkai and LeCun, Yann and Pinto, Lerrel},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/zhou25t.html}
}
@misc{assran2025vjepa2,
  title         = {{V-JEPA} 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author        = {Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and others},
  year          = {2025},
  eprint        = {2506.09985},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2506.09985}
}
@inproceedings{dedieu2025improving,
  title     = {Improving Transformer World Models for Data-Efficient {RL}},
  author    = {Dedieu, Antoine and Ortiz, Joseph and Lou, Xinghua and Wendelken, Carter and Lehrach, Wolfgang and Guntupalli, J. Swaroop and L{\'a}zaro-Gredilla, Miguel and Murphy, Kevin Patrick},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/dedieu25a.html}
}
@inproceedings{vafa2024evaluating,
  title     = {Evaluating the World Model Implicit in a Generative Model},
  author    = {Vafa, Keyon and Chen, Justin Y. and Rambachan, Ashesh and Kleinberg, Jon and Mullainathan, Sendhil},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024},
  eprint    = {2406.03689},
  archivePrefix = {arXiv}
}
@inproceedings{motamed2026physicsiq,
  title     = {Do generative video models understand physical principles?},
  author    = {Motamed, Saman and Culp, Laura and Swersky, Kevin and Jaini, Priyank and Geirhos, Robert},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {948--958},
  year      = {2026},
  eprint    = {2501.09038},
  archivePrefix = {arXiv}
}
@inproceedings{joseph2026physics,
  title     = {Interpreting Physics in Video World Models},
  author    = {Joseph, Sonia and Garrido, Quentin and Balestriero, Randall and Kowal, Matthew and Fel, Thomas and Bakhtiari, Shahab and Richards, Blake Aaron and Rabbat, Michael},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  pages     = {54687--54724},
  publisher = {PMLR},
  year      = {2026},
  url       = {https://proceedings.mlr.press/v306/joseph26a.html}
}
@inproceedings{escoriza2025demo3,
  title     = {Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning},
  author    = {Escoriza, Adri{\`a} L{\'o}pez and Hansen, Nicklas and Tao, Stone and Mu, Tongzhou and Su, Hao},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/escoriza25a.html}
}
@inproceedings{paglieri2025balrog,
  title     = {{BALROG}: Benchmarking Agentic {LLM} and {VLM} Reasoning On Games},
  author    = {Paglieri, Davide and Cupia{\l}, Bart{\l}omiej and Coward, Samuel and Piterbarg, Ulyana and others},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html}
}
"""


def main() -> None:
    text = BIB.read_text(encoding="utf-8")
    entries = re.split(r"(?=^@)", text, flags=re.M)
    kept = []
    for chunk in entries:
        m = re.match(r"@\w+\{([^,]+),", chunk)
        if m and m.group(1).strip() in DROP:
            tail = re.search(r"\n(%[^\n]*\n)\s*$", chunk)
            kept.append(tail.group(1) if tail else "")
            continue
        kept.append(chunk)
    out = "".join(kept)
    out = out.replace(
        "  primaryClass  = {cs.RO}\n}",
        "  primaryClass  = {cs.RO},\n  note          = {ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling}\n}",
    )
    BIB.write_text(out.rstrip() + "\n" + NEW, encoding="utf-8")


if __name__ == "__main__":
    main()
