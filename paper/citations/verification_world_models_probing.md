I'm noting that the Readable Coordinates paper is an anonymous ICLR 2027 submission under review, so I can't cite it with authors and will flag it as not citable yet. Next I'm checking the remaining details for the output, starting with the STORM and DIAMOND DOI values from the NeurIPS proceedings page.

I'm working through the citation details now: the STORM and DIAMOND entries can include their DOIs and pages from the NeurIPS proceedings, while AtariARI and Ha will go without pages because the sources disagree or don't show them. For the rest of the list I'm pulling the volume and page numbers for PlaNet, DreamerPro, MWM, Plan2Explore, TIA, and HarmonyDream, and checking Dyna's DOI against the ACM page.

I'm finishing the Nature entry for DreamerV3, where the DOI was seen on nature.com under the title "Mastering diverse control tasks through world models," so I'll list it alongside the arXiv version. For Dreaming (ICRA 2021), I'm leaving off the DOI and pages, since the IEEE page didn't render and only the Semantic Scholar page shows them, so I'll mention them as unconfirmed. For Hewitt's EMNLP-IJCNLP 2019 paper, I'm keeping the pages 2733–2743 from the ACL page and dropping the DOI, which only appeared in Semantic Scholar.

I'm filling in the remaining citation details: the Hewitt pages and the omitted DOI, then the Cho pages and DOI from the ACL page. Next come the arXiv-style venue entries for the Informed POMDP, Scaffolder, TD-MPC2, Dreamer, DreamerV2, and SimPLe papers, with their OpenReview IDs.

I'm going through the remaining venue details: the IRIS entry is ICLR 2023 as an oral, and I'm adding its forum ID from the GitHub README and iclr.cc. Othello also goes in as an ICLR 2023 oral with its forum ID. Alain is listed as an ICLR 2017 workshop paper, but the forum IDs conflict, so I'll use the arXiv link for it and mark it as only partly verified.

I'm listing the remaining citation entries, including the Dreamer 4 arXiv record, and I'm drafting the candidate items, starting with the Gonzalez Laiz and Schneider 2026 paper that probes DreamerV3-style latents for physical simulator state.

I'm noting a few more candidate papers to list: Vakalis 2026 trains a DreamerV3 RSSM with an auxiliary head and a linear probe in a synthetic environment, which bears on both items 29 and 30. I'm also flagging the "Readable Coordinates" ICLR 2027 submission and the dreamerv3f GitHub repo, which isn't a paper. My conclusion so far is that I haven't found a peer-reviewed paper probing symbolic Crafter or Atari tile and object state from DreamerV2 or V3 RSSM latents.

I'm listing the remaining candidates for item 30, starting with Krinner et al. 2025, where a privileged decoder rebuilds hidden state inside a DreamerV3-style world model, and PrivilegedDreamer, which estimates hidden parameters with a reconstruction loss on the RSSM latents. I'm also noting Reinformed Dreamer for its asymmetric latent guidance and MILE, which decodes BEV segmentation from its world model's latent, though that one isn't strictly a Dreamer variant.

I'm narrowing the item-30 list to three, since there are four candidates and the cap is three. I'll keep Krinner et al. and PrivilegedDreamer, since PrivilegedDreamer estimates hidden parameters and fits the "privileged" framing best, and add MILE for its decoded bird's-eye-view semantic segmentation. Reinformed Dreamer I'll mention as a follow-up to item 19, since it's the closest match and shares authors with that item.

I'm settling the last picks: three candidates each for items 29 and 30, with Reinformed Dreamer and MILE in for item 30 and PrivilegedDreamer as a note. I'll also add Crafter as a bonus entry, since it documents the semantic map.

I'm moving on to the BibTeX entries now, with SimPLe and MuZero shortened to first author plus "and others" and MILE likewise, while the rest get their full author lists.

I'm now checking the PMLR volume numbers for the remaining entries, and I'm deciding whether MWM should be listed as 2022 or 2023, since its PMLR proceedings were published in March 2023 even though the conference was in 2022.

I'm laying out the table's columns as key, verified status, contribution, and unconfirmed details, with the Hewitt booktitle as the final citation field to settle.


All 28 named papers exist with the expected titles, authors and arXiv IDs. I checked each one against arXiv's metadata and the proceedings or publisher page.

DreamerV3 did appear in Nature (2025), but under a different title: "Mastering diverse control tasks through world models." I'd cite both the Nature version and the arXiv preprint.

For items 29 and 30 I found no peer-reviewed paper that probes symbolic Crafter/Atari state from a DreamerV2/V3 latent. I also found none that adds a symbolic-map prediction head to a Dreamer world model. The nearest verified candidates are listed below; most are arXiv-only from 2025–2026.

Page numbers and DOIs below appear only where I saw them on a proceedings or publisher page. Where they came from elsewhere, the table says where.

## A) BibTeX

```bibtex
% ---------- 1. World Models ----------
@misc{ha2018worldmodels,
  title         = {World Models},
  author        = {Ha, David and Schmidhuber, J{\"u}rgen},
  year          = {2018},
  eprint        = {1803.10122},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1803.10122}
}
@inproceedings{ha2018recurrent,
  title     = {Recurrent World Models Facilitate Policy Evolution},
  author    = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018},
  url       = {https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html}
}

% ---------- 2. PlaNet ----------
@inproceedings{hafner2019planet,
  title     = {Learning Latent Dynamics for Planning from Pixels},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {2555--2565},
  publisher = {PMLR},
  year      = {2019},
  eprint    = {1811.04551},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v97/hafner19a.html}
}

% ---------- 3. Dreamer ----------
@inproceedings{hafner2020dreamer,
  title     = {Dream to Control: Learning Behaviors by Latent Imagination},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  eprint    = {1912.01603},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=S1lOTC4tDS}
}

% ---------- 4. DreamerV2 ----------
@inproceedings{hafner2021dreamerv2,
  title     = {Mastering Atari with Discrete World Models},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  eprint    = {2010.02193},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=0oabwyZbOu}
}

% ---------- 5. DreamerV3 (arXiv + Nature) ----------
@misc{hafner2023dreamerv3,
  title         = {Mastering Diverse Domains through World Models},
  author        = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  year          = {2023},
  eprint        = {2301.04104},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2301.04104}
}
@article{hafner2025dreamerv3nature,
  title   = {Mastering diverse control tasks through world models},
  author  = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal = {Nature},
  volume  = {640},
  number  = {8059},
  pages   = {647--653},
  year    = {2025},
  doi     = {10.1038/s41586-025-08744-2},
  url     = {https://www.nature.com/articles/s41586-025-08744-2}
}

% ---------- 6. Dreamer 4 ----------
@misc{hafner2025dreamer4,
  title         = {Training Agents Inside of Scalable World Models},
  author        = {Hafner, Danijar and Yan, Wilson and Lillicrap, Timothy},
  year          = {2025},
  eprint        = {2509.24527},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2509.24527}
}

% ---------- 7. Dyna ----------
@article{sutton1991dyna,
  title   = {Dyna, an Integrated Architecture for Learning, Planning, and Reacting},
  author  = {Sutton, Richard S.},
  journal = {ACM SIGART Bulletin},
  volume  = {2},
  number  = {4},
  pages   = {160--163},
  year    = {1991},
  doi     = {10.1145/122344.122377},
  url     = {https://dl.acm.org/doi/10.1145/122344.122377}
}

% ---------- 8. MuZero ----------
@article{schrittwieser2020muzero,
  title   = {Mastering {Atari}, {Go}, chess and shogi by planning with a learned model},
  author  = {Schrittwieser, Julian and Antonoglou, Ioannis and Hubert, Thomas and Simonyan, Karen and Sifre, Laurent and Schmitt, Simon and Guez, Arthur and Lockhart, Edward and Hassabis, Demis and Graepel, Thore and Lillicrap, Timothy and Silver, David},
  journal = {Nature},
  volume  = {588},
  number  = {7839},
  pages   = {604--609},
  year    = {2020},
  doi     = {10.1038/s41586-020-03051-4},
  eprint  = {1911.08265},
  archivePrefix = {arXiv},
  url     = {https://www.nature.com/articles/s41586-020-03051-4}
}

% ---------- 9. SimPLe ----------
@inproceedings{kaiser2020simple,
  title     = {Model Based Reinforcement Learning for {Atari}},
  author    = {Kaiser, {\L}ukasz and others},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  eprint    = {1903.00374},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=S1xCPJHtDB}
}

% ---------- 10. IRIS ----------
@inproceedings{micheli2023iris,
  title     = {Transformers are Sample-Efficient World Models},
  author    = {Micheli, Vincent and Alonso, Eloi and Fleuret, Fran{\c{c}}ois},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  eprint    = {2209.00588},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=vhFu1Acb0xb}
}

% ---------- 11. STORM ----------
@inproceedings{zhang2023storm,
  title     = {{STORM}: Efficient Stochastic Transformer based World Models for Reinforcement Learning},
  author    = {Zhang, Weipu and Wang, Gang and Sun, Jian and Yuan, Yetian and Huang, Gao},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {27147--27166},
  year      = {2023},
  doi       = {10.52202/075280-1182},
  eprint    = {2310.09615},
  archivePrefix = {arXiv},
  url       = {https://papers.nips.cc/paper_files/paper/2023/hash/5647763d4245b23e6a1cb0a8947b38c9-Abstract-Conference.html}
}

% ---------- 12. DIAMOND ----------
@inproceedings{alonso2024diamond,
  title     = {Diffusion for World Modeling: Visual Details Matter in {Atari}},
  author    = {Alonso, Eloi and Jelley, Adam and Micheli, Vincent and Kanervisto, Anssi and Storkey, Amos and Pearce, Tim and Fleuret, Fran{\c{c}}ois},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {58757--58791},
  year      = {2024},
  doi       = {10.52202/079017-1873},
  eprint    = {2405.12399},
  archivePrefix = {arXiv},
  url       = {https://papers.nips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html}
}

% ---------- 13. TD-MPC2 ----------
@inproceedings{hansen2024tdmpc2,
  title     = {{TD-MPC2}: Scalable, Robust World Models for Continuous Control},
  author    = {Hansen, Nicklas and Su, Hao and Wang, Xiaolong},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  eprint    = {2310.16828},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=Oxh5CstDJU}
}

% ---------- 14. DreamerPro ----------
@inproceedings{deng2022dreamerpro,
  title     = {{DreamerPro}: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations},
  author    = {Deng, Fei and Jang, Ingook and Ahn, Sungjin},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {4956--4975},
  publisher = {PMLR},
  year      = {2022},
  eprint    = {2110.14565},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v162/deng22a.html}
}

% ---------- 15. Dreaming ----------
@inproceedings{okada2021dreaming,
  title     = {Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction},
  author    = {Okada, Masashi and Taniguchi, Tadahiro},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2021},
  eprint    = {2007.14535},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2007.14535}
}

% ---------- 16. MWM ----------
@inproceedings{seo2023mwm,
  title     = {Masked World Models for Visual Control},
  author    = {Seo, Younggyo and Hafner, Danijar and Liu, Hao and Liu, Fangchen and James, Stephen and Lee, Kimin and Abbeel, Pieter},
  booktitle = {Proceedings of The 6th Conference on Robot Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {205},
  pages     = {1332--1344},
  publisher = {PMLR},
  year      = {2023},
  eprint    = {2206.14244},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v205/seo23a.html}
}

% ---------- 17. TIA ----------
@inproceedings{fu2021tia,
  title     = {Learning Task Informed Abstractions},
  author    = {Fu, Xiang and Yang, Ge and Agrawal, Pulkit and Jaakkola, Tommi},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {3480--3491},
  publisher = {PMLR},
  year      = {2021},
  eprint    = {2106.15612},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v139/fu21b.html}
}

% ---------- 18. HarmonyDream ----------
@inproceedings{ma2024harmonydream,
  title     = {{HarmonyDream}: Task Harmonization Inside World Models},
  author    = {Ma, Haoyu and Wu, Jialong and Feng, Ningya and Xiao, Chenjun and Li, Dong and Hao, Jianye and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {33983--34007},
  publisher = {PMLR},
  year      = {2024},
  eprint    = {2310.00344},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v235/ma24o.html}
}

% ---------- 19. Informed POMDP ----------
@article{lambrechts2024informed,
  title   = {Informed {POMDP}: Leveraging Additional Information in Model-Based {RL}},
  author  = {Lambrechts, Gaspard and Bolland, Adrien and Ernst, Damien},
  journal = {Reinforcement Learning Journal},
  volume  = {2},
  pages   = {763--784},
  year    = {2024},
  note    = {Presented at the Reinforcement Learning Conference (RLC) 2024},
  eprint  = {2306.11488},
  archivePrefix = {arXiv},
  url     = {https://rlj.org/papers/lambrechts2024informed}
}

% ---------- 20. Scaffolder ----------
@inproceedings{hu2024scaffolder,
  title     = {Privileged Sensing Scaffolds Reinforcement Learning},
  author    = {Hu, Edward S. and Springer, James and Rybkin, Oleh and Jayaraman, Dinesh},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  eprint    = {2405.14853},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=EpVe8jAjdx}
}

% ---------- 21. Linear probes ----------
@inproceedings{alain2017probes,
  title     = {Understanding intermediate layers using linear classifier probes},
  author    = {Alain, Guillaume and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations (Workshop Track)},
  year      = {2017},
  eprint    = {1610.01644},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1610.01644}
}

% ---------- 22. Control tasks ----------
@inproceedings{hewitt2019control,
  title     = {Designing and Interpreting Probes with Control Tasks},
  author    = {Hewitt, John and Liang, Percy},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {2733--2743},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  eprint    = {1909.03368},
  archivePrefix = {arXiv},
  url       = {https://aclanthology.org/D19-1275/}
}

% ---------- 23. AtariARI ----------
@inproceedings{anand2019atariari,
  title     = {Unsupervised State Representation Learning in {Atari}},
  author    = {Anand, Ankesh and Racah, Evan and Ozair, Sherjil and Bengio, Yoshua and C{\^o}t{\'e}, Marc-Alexandre and Hjelm, R Devon},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019},
  eprint    = {1906.08226},
  archivePrefix = {arXiv},
  url       = {https://papers.nips.cc/paper_files/paper/2019/hash/6fb52e71b837628ac16539c1ff911667-Abstract.html}
}

% ---------- 24. Othello-GPT ----------
@inproceedings{li2023othello,
  title     = {Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task},
  author    = {Li, Kenneth and Hopkins, Aspen K. and Bau, David and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  eprint    = {2210.13382},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=DeG07_TcZvT}
}

% ---------- 25. VAE ----------
@inproceedings{kingma2014vae,
  title     = {Auto-Encoding Variational {Bayes}},
  author    = {Kingma, Diederik P. and Welling, Max},
  booktitle = {International Conference on Learning Representations},
  year      = {2014},
  eprint    = {1312.6114},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1312.6114}
}

% ---------- 26. Straight-through ----------
@misc{bengio2013estimating,
  title         = {Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation},
  author        = {Bengio, Yoshua and L{\'e}onard, Nicholas and Courville, Aaron},
  year          = {2013},
  eprint        = {1308.3432},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1308.3432}
}

% ---------- 27. GRU ----------
@inproceedings{cho2014gru,
  title     = {Learning Phrase Representations using {RNN} Encoder--Decoder for Statistical Machine Translation},
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {1724--1734},
  publisher = {Association for Computational Linguistics},
  address   = {Doha, Qatar},
  year      = {2014},
  doi       = {10.3115/v1/D14-1179},
  eprint    = {1406.1078},
  archivePrefix = {arXiv},
  url       = {https://aclanthology.org/D14-1179/}
}

% ---------- 28. Plan2Explore ----------
@inproceedings{sekar2020plan2explore,
  title     = {Planning to Explore via Self-Supervised World Models},
  author    = {Sekar, Ramanan and Rybkin, Oleh and Daniilidis, Kostas and Abbeel, Pieter and Hafner, Danijar and Pathak, Deepak},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {8583--8592},
  publisher = {PMLR},
  year      = {2020},
  eprint    = {2005.05960},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v119/sekar20a.html}
}

% ---------- 29. Candidates: probing world-model latents for state ----------
@misc{gonzalezlaiz2026limits,
  title         = {Understanding the limits of latent world models},
  author        = {Gonzalez Laiz, Rodrigo and Schneider, Steffen},
  year          = {2026},
  eprint        = {2610.37588},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.37588}
}
@misc{vakalis2026threequestions,
  title         = {What a World Model Represents Is Three Questions},
  author        = {Vakalis, Donna},
  year          = {2026},
  eprint        = {2607.06640},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.06640}
}
@misc{zhang2026probing,
  title         = {What Do World Models Learn in {RL}? Probing Latent Representations in Learned Environment Simulators},
  author        = {Zhang, Xinyu},
  year          = {2026},
  eprint        = {2603.21546},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {ICLR 2026 Workshop on World Models: Understanding, Modelling and Scaling},
  url           = {https://arxiv.org/abs/2603.21546}
}

% ---------- 30. Candidates: privileged/semantic prediction in Dreamer-style WMs ----------
@misc{lambrechts2026reinformed,
  title         = {Reinformed Dreamer: An Asymmetric World Model Efficiently Trained through Latent Guidance},
  author        = {Lambrechts, Gaspard and Bolland, Adrien and Ebi, Daniel and Ernst, Damien},
  year          = {2026},
  eprint        = {2607.26040},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.26040}
}
@misc{krinner2025ssmwm,
  title         = {Accelerating Model-Based Reinforcement Learning with State-Space World Models},
  author        = {Krinner, Maria and Aljalbout, Elie and Romero, Angel and Scaramuzza, Davide},
  year          = {2025},
  eprint        = {2502.20168},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2502.20168}
}
@inproceedings{hu2022mile,
  title     = {Model-Based Imitation Learning for Urban Driving},
  author    = {Hu, Anthony and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  eprint    = {2210.07729},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2210.07729}
}

% ---------- Extra (not requested; this is the benchmark you train on) ----------
@inproceedings{hafner2022crafter,
  title     = {Benchmarking the Spectrum of Agent Capabilities},
  author    = {Hafner, Danijar},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
  eprint    = {2109.06780},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2109.06780}
}
```

## B) Verification table

| key | verified? | relevant contribution (from abstract) | could not confirm / notes |
|---|---|---|---|
| ha2018worldmodels | yes | Trains a generative model of RL environments without supervision, giving a compressed spatial and temporal representation; a compact policy trained on its features can solve the task, even when trained inside the model's own "dream". | The arXiv record carries Zenodo DOI 10.5281/zenodo.1207631; omitted because I saw it only in arXiv metadata. |
| ha2018recurrent | yes | The NeurIPS 2018 version of World Models: a recurrent generative world model whose features support policies evolved inside the model. | Pages not shown on the NeurIPS page (Nature's reference list gives 2455–2467 for the NIPS'18 proceedings); omitted. |
| hafner2019planet | yes | Introduces PlaNet and the RSSM: a latent dynamics model with deterministic and stochastic parts, used for online planning purely in latent space from pixels. | — |
| hafner2020dreamer | yes | Learns behaviours purely by latent imagination, backpropagating analytic value gradients through imagined trajectories of a learned world model. | Venue confirmed via an OpenReview ICLR 2020 record; no pages exist. |
| hafner2021dreamerv2 | yes | Discrete (categorical) latents in the RSSM let a world-model agent reach human-level Atari performance by learning only from imagined trajectories. | OpenReview lists it as an ICLR 2021 poster. |
| hafner2023dreamerv3 | yes | A single fixed configuration across 150+ tasks, stabilised by normalisation, balancing and transformations; first to collect Minecraft diamonds from scratch. | — |
| hafner2025dreamerv3nature | yes | Same work, published in Nature 2025 under the title "Mastering diverse control tasks through world models". | The DOI was seen on nature.com. Volume 640, issue 8059 and pages 647–653 came from PubMed, not the Nature page. |
| hafner2025dreamer4 | yes | Dreamer 4 learns control by RL inside a fast, accurate, scalable video world model that better predicts object interactions in complex environments. | arXiv only; no venue found. |
| sutton1991dyna | yes | Dyna combines learning, planning with a learned model, and reactive execution: planning is "trying things in your head" with simulated experience. | The ACM page confirmed volume 2, issue 4 (Aug 1991) and the DOI. Pages 160–163 came from Google Scholar and Semantic Scholar. |
| schrittwieser2020muzero | yes | Plans with tree search over a learned model that predicts only policy, value and reward, matching AlphaZero without knowing the rules. | Pages and issue are from RePEc/Europe PMC and the reference list in Nature's DreamerV3 article. |
| kaiser2020simple | yes | SimPLe trains policies inside learned video-prediction models and beats model-free methods at 100k Atari interactions. | 14 authors, so "and others". OpenReview title is "Model Based…" (arXiv says "Model-Based…"). |
| micheli2023iris | yes | IRIS uses a discrete autoencoder plus an autoregressive Transformer as the world model, reaching 1.046 mean human-normalised score on Atari 100k. | Forum ID seen on the iclr.cc page and the authors' GitHub, not on OpenReview directly. |
| zhang2023storm | yes | A stochastic Transformer world model that pairs categorical-VAE latents with a Transformer sequence model for sample-efficient Atari 100k. | — |
| alonso2024diamond | yes | A diffusion-based world model that keeps the visual detail discrete-latent models compress away, which matters for RL on Atari. | NeurIPS 2024 spotlight, per the arXiv comment. |
| hansen2024tdmpc2 | yes | A decoder-free implicit world model with local trajectory optimisation that scales across 104 continuous-control tasks. | ICLR 2024 spotlight. |
| deng2022dreamerpro | yes | Replaces Dreamer's image reconstruction with prototypical representations, making it robust to visual distractions. | — |
| okada2021dreaming | partially | A decoder-free Dreamer using a contrastive InfoMax objective, to stop task-relevant objects "vanishing" under reconstruction. | ICRA 2021 is confirmed by arXiv and the DOI landing page. The IEEE page did not render, so DOI 10.1109/ICRA48506.2021.9560734 and pages 4209–4215 (Semantic Scholar) are omitted. |
| seo2023mwm | yes | Separates visual representation learning (a masked autoencoder) from dynamics learning, improving visual model-based RL. | CoRL 2022 proceedings (PMLR 205) are dated March 2023; I used year 2023. Change to 2022 if you prefer the conference year. |
| fu2021tia | yes | Splits the latent into reward-correlated and distractor components, so the model prioritises task-relevant features. | — |
| ma2024harmonydream | yes | Shows that observation modelling can dominate reward modelling inside world models, and dynamically re-weights the two losses. | — |
| lambrechts2024informed | yes | Formalises training-time-only information and shows that predicting it, rather than observations, in Dreamer's world model speeds learning. **This is your closest prior work.** | — |
| hu2024scaffolder | yes | Uses privileged training-only sensors to scaffold the world model, critic and other components, while the deployed policy sees only target observations. | ICLR 2024 spotlight. |
| alain2017probes | partially | Introduces linear classifier "probes", trained independently of the model, to measure how linearly decodable each layer's features are. | ICLR 2017 Workshop Track is from researchr and ML Anthology. Sources disagree on the OpenReview forum ID, so the url points to arXiv. |
| hewitt2019control | yes | Introduces control tasks and "selectivity", so probe accuracy reflects the representation rather than the probe's own capacity. | DOI 10.18653/v1/D19-1275 (from Semantic Scholar) not seen on the ACL page; omitted. |
| anand2019atariari | yes | Introduces the AtariARI benchmark, which supplies RAM-derived ground-truth state labels for probing learned Atari representations. | Pages not shown on the NeurIPS page. |
| li2023othello | yes | Shows a sequence model trained only on Othello moves develops an internal board representation that probes can recover and interventions can edit. | ICLR 2023 oral (notable top 5%). |
| kingma2014vae | yes | The reparameterised variational lower bound (SGVB) and the AEVB / variational autoencoder. | Listed in the ICLR 2014 conference proceedings archive; the url points to arXiv. |
| bengio2013estimating | yes | Compares gradient estimators for stochastic or hard neurons, including the straight-through estimator Dreamer uses for categorical latents. | arXiv only. |
| cho2014gru | yes | The RNN Encoder–Decoder, which introduced the gated recurrent unit (GRU) used in the RSSM. | — |
| sekar2020plan2explore | yes | A self-supervised world-model agent that explores by planning toward expected future novelty (ensemble disagreement), then adapts zero- or few-shot to new tasks. | — |
| gonzalezlaiz2026limits | partially (item 29) | Probes frozen DreamerV3, TD-MPC2 and LeWM latents for simulator state and finds that none faithfully models the dynamics. | arXiv only. It probes *physical* state in Pendulum, CartPole and Reacher, not symbolic game state. |
| vakalis2026threequestions | partially (items 29/30) | On a DreamerV3 RSSM, adds an auxiliary head on the stochastic latent and measures what it installs with linear probes. | arXiv only. Uses a synthetic environment, not Crafter or Atari. The v1 title was "The Rank-One Corner…". |
| zhang2026probing | partially (item 29) | Linear and MLP probes plus causal interventions recover Atari object positions and scores from IRIS and DIAMOND internals. | Workshop paper; IRIS and DIAMOND are not RSSMs. |
| lambrechts2026reinformed | partially (item 30) | Finds a limitation in the Informed Dreamer's privileged-information representations and proposes latent guidance instead. | arXiv only, Jul 2026. |
| krinner2025ssmwm | partially (item 30) | In a DreamerV3-style model with a state-space-model sequence model, replaces the image decoder with a decoder that reconstructs privileged simulator state. | arXiv only; no venue found. The privileged state is quadrotor physical state, not symbolic. |
| hu2022mile | partially (item 30) | MILE's driving world model learns a compact latent whose predicted states can be decoded to bird's-eye-view *semantic segmentation*. | NeurIPS 2022 is only from the arXiv comment; not seen on a proceedings page. The model is RSSM-like, not Dreamer itself. 9 authors, so "and others". |
| hafner2022crafter | yes (extra) | The Crafter benchmark; it documents privileged information including a semantic grid map, intended for evaluating representations. | ICLR 2022 is from the arXiv comment only. |

### What I found for items 29 and 30

- **Item 29:** No peer-reviewed paper decodes symbolic Crafter or Atari tile/object state from a DreamerV2/V3 RSSM latent. Two things exist but shouldn't go in the bibliography:
  - **Informal DreamerV2 probe:** a blog post (Mu Lab) probes a discretised DreamerV2 for Pong ball position and velocity. It is not a paper.
  - **ICLR 2027 submission:** "Readable Coordinates, Distributed Concepts: Concept Supervision in DreamerV3" supervises 14 Craftax-Classic concepts on DreamerV3 latent coordinates. It is directly relevant, but it is anonymous and under review, so I left it out. Re-check after decisions are released.
  - A GitHub fork (`vviggyy/dreamerv3f`) decodes player position from DreamerV3 latents in Crafter, but has no paper.
- **Item 30:** No paper adds a symbolic *map* head inside a Dreamer world model. The verified candidates above predict either physical privileged state (the quadrotor paper, Reinformed Dreamer) or semantic bird's-eye-view maps in a non-Dreamer RSSM-like model (MILE).
  - PrivilegedDreamer (Byrd et al., arXiv 2502.11377, ICRA 2025 per its arXiv comment) is another verified option. It adds an explicit module that estimates hidden parameters, but they are dynamics parameters rather than symbolic state.