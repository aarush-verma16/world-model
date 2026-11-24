I've confirmed the result, so the next step is writing the final summary. The answer should stay concise but complete, and since no files are needed, I'll deliver it through UpdateCurrentStep.


All 27 named papers exist and match on title, authors and venue. None are fabricated. Six of the citations you gave need fixing:

- **Moon et al.** has four authors: Bumsoo Park is missing from "Moon, Yeom, Song".
- **SPRING**'s NeurIPS proceedings title is "SPRING: Studying Papers and Reasoning to play Games". The author order also differs from arXiv.
- **ALVINN** is listed under 1988 in Advances in Neural Information Processing Systems 1.
- **Stanić et al.** is IEEE Transactions on Games, 2023.
- **Learning by Cheating**: the official PMLR BibTeX gives year 2020 for CoRL 2019.
- **Cognitive Mapping**: the CVPR 2017 paper has five authors. The arXiv v3 is the longer IJCV version and adds Varun Tolani.

OpenReview returned "403 Forbidden" to my fetcher. For the ICLR papers I confirmed the venue through iclr.cc, the arXiv "Published at ICLR" comments, the papers' own text, and indexes (ML Anthology, researchr). The BibTeX URLs therefore point to arXiv. I only included a DOI where I saw it on a publisher or proceedings page. DOIs I saw only through doi.org lookups or secondary indexes are listed in table B and left out of the BibTeX.

## A) BibTeX

```bibtex
% ---------- Crafter and agents on it ----------
@inproceedings{hafner2022crafter,
  title     = {Benchmarking the Spectrum of Agent Capabilities},
  author    = {Hafner, Danijar},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  eprint    = {2109.06780}, archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2109.06780}
}
@inproceedings{matthews2024craftax,
  title     = {Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  author    = {Matthews, Michael and Beukman, Michael and Ellis, Benjamin and Samvelyan, Mikayel and Jackson, Matthew Thomas and Coward, Samuel and Foerster, Jakob Nicolaus},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {235}, pages = {35104--35137},
  year      = {2024},
  eprint    = {2402.16801}, archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v235/matthews24a.html}
}
@article{stanic2023crafterood,
  title   = {Learning to Generalize With Object-Centric Agents in the Open World Survival Game Crafter},
  author  = {Stani{\'c}, Aleksandar and Tang, Yujin and Ha, David and Schmidhuber, J{\"u}rgen},
  journal = {IEEE Transactions on Games},
  year    = {2023},
  eprint  = {2208.03374}, archivePrefix = {arXiv},
  url     = {https://ieeexplore.ieee.org/document/10125026/}
}
@inproceedings{walker2023modelbased,
  title     = {Investigating the Role of Model-Based Learning in Exploration and Transfer},
  author    = {Walker, Jacob C and V{\'e}rtes, Eszter and Li, Yazhe and Dulac-Arnold, Gabriel and Anand, Ankesh and Weber, Theophane and Hamrick, Jessica B},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {202}, pages = {35368--35383},
  year      = {2023},
  eprint    = {2302.04009}, archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v202/walker23a.html}
}
@inproceedings{moon2023achievement,
  title     = {Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning},
  author    = {Moon, Seungyong and Yeom, Junyoung and Park, Bumsoo and Song, Hyun Oh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2307.03486}, archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/c919a2b5ec1de69f2629f9119676e336-Abstract-Conference.html}
}
@inproceedings{hafner2022director,
  title     = {Deep Hierarchical Planning from Pixels},
  author    = {Hafner, Danijar and Lee, Kuang-Huei and Fischer, Ian and Abbeel, Pieter},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  eprint    = {2206.04114}, archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html}
}
@inproceedings{du2023ellm,
  title     = {Guiding Pretraining in Reinforcement Learning with Large Language Models},
  author    = {Du, Yuqing and Watkins, Olivia and Wang, Zihan and Colas, C{\'e}dric and Darrell, Trevor and Abbeel, Pieter and Gupta, Abhishek and Andreas, Jacob},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {202}, pages = {8657--8677},
  year      = {2023},
  eprint    = {2302.06692}, archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v202/du23f.html}
}
@inproceedings{wu2023spring,
  title     = {{SPRING}: Studying Papers and Reasoning to play Games},
  author    = {Wu, Yue and Min, So Yeon and Prabhumoye, Shrimai and Bisk, Yonatan and Salakhutdinov, Ruslan and Azaria, Amos and Mitchell, Tom M. and Li, Yuanzhi},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS)},
  year      = {2023},
  doi       = {10.52202/075280-0983},
  eprint    = {2305.15486}, archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/46c2a9a6f2b2be68682013eb1173c801-Abstract-Conference.html}
}

% ---------- Demonstrations / imitation ----------
@inproceedings{pomerleau1988alvinn,
  title     = {{ALVINN}: An Autonomous Land Vehicle in a Neural Network},
  author    = {Pomerleau, Dean A.},
  booktitle = {Advances in Neural Information Processing Systems 1 (NIPS 1988)},
  editor    = {Touretzky, D. S.},
  pages     = {305--313},
  year      = {1988},
  url       = {https://papers.nips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html}
}
@inproceedings{ross2011dagger,
  title     = {A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning},
  author    = {Ross, St{\'e}phane and Gordon, Geoffrey J. and Bagnell, J. Andrew},
  booktitle = {Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {PMLR}, volume = {15},
  year      = {2011},
  eprint    = {1011.0686}, archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v15/ross11a.html}
}
@inproceedings{hester2018dqfd,
  title     = {Deep {Q}-learning from Demonstrations},
  author    = {Hester, Todd and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32}, number = {1},
  year      = {2018},
  doi       = {10.1609/aaai.v32i1.11757},
  eprint    = {1704.03732}, archivePrefix = {arXiv},
  url       = {https://aaai.org/papers/11757-deep-q-learning-from-demonstrations/}
}
@inproceedings{nair2018overcoming,
  title     = {Overcoming Exploration in Reinforcement Learning with Demonstrations},
  author    = {Nair, Ashvin and McGrew, Bob and Andrychowicz, Marcin and Zaremba, Wojciech and Abbeel, Pieter},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2018},
  eprint    = {1709.10089}, archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1709.10089}
}
@inproceedings{rajeswaran2018dapg,
  title     = {Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations},
  author    = {Rajeswaran, Aravind and Kumar, Vikash and Gupta, Abhishek and Vezzani, Giulia and Schulman, John and Todorov, Emanuel and Levine, Sergey},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2018},
  doi       = {10.15607/RSS.2018.XIV.049},
  eprint    = {1709.10087}, archivePrefix = {arXiv},
  url       = {https://www.roboticsproceedings.org/rss14/p49.html}
}
@inproceedings{baker2022vpt,
  title     = {Video PreTraining ({VPT}): Learning to Act by Watching Unlabeled Online Videos},
  author    = {Baker, Bowen and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  eprint    = {2206.11795}, archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html}
}
@inproceedings{guss2019minerl,
  title     = {{MineRL}: A Large-Scale Dataset of {Minecraft} Demonstrations},
  author    = {Guss, William H. and Houghton, Brandon and Topin, Nicholay and Wang, Phillip and Codel, Cayden and Veloso, Manuela and Salakhutdinov, Ruslan},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)},
  pages     = {2442--2448},
  year      = {2019},
  doi       = {10.24963/ijcai.2019/339},
  eprint    = {1907.13440}, archivePrefix = {arXiv},
  url       = {https://ijcai.org/proceedings/2019/339}
}
@article{ecoffet2021firstreturn,
  title   = {First return, then explore},
  author  = {Ecoffet, Adrien and Huizinga, Joost and Lehman, Joel and Stanley, Kenneth O. and Clune, Jeff},
  journal = {Nature},
  volume  = {590}, pages = {580--586},
  year    = {2021},
  doi     = {10.1038/s41586-020-03157-9},
  eprint  = {2004.12919}, archivePrefix = {arXiv},
  url     = {https://www.nature.com/articles/s41586-020-03157-9}
}
@misc{ecoffet2019goexplore,
  title  = {Go-Explore: a New Approach for Hard-Exploration Problems},
  author = {Ecoffet, Adrien and Huizinga, Joost and Lehman, Joel and Stanley, Kenneth O. and Clune, Jeff},
  year   = {2019},
  eprint = {1901.10995}, archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/1901.10995}
}
@inproceedings{andrychowicz2017her,
  title     = {Hindsight Experience Replay},
  author    = {Andrychowicz, Marcin and others},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  year      = {2017},
  eprint    = {1707.01495}, archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf}
}
@article{argall2009survey,
  title   = {A survey of robot learning from demonstration},
  author  = {Argall, Brenna D. and Chernova, Sonia and Veloso, Manuela and Browning, Brett},
  journal = {Robotics and Autonomous Systems},
  volume  = {57}, pages = {469--483},
  year    = {2009},
  url     = {https://www.cs.cmu.edu/~mmv/papers/09ras-survey.pdf}
}

% ---------- Privileged information ----------
@article{vapnik2009lupi,
  title   = {A new learning paradigm: Learning using privileged information},
  author  = {Vapnik, Vladimir and Vashist, Akshay},
  journal = {Neural Networks},
  volume  = {22}, pages = {544--557},
  year    = {2009},
  url     = {https://pubmed.ncbi.nlm.nih.gov/19632812/}
}
@inproceedings{chen2019lbc,
  title     = {Learning by Cheating},
  author    = {Chen, Dian and Zhou, Brady and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle = {Proceedings of the Conference on Robot Learning (CoRL 2019)},
  series    = {PMLR}, volume = {100}, pages = {66--75},
  year      = {2020},
  eprint    = {1912.12294}, archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v100/chen20a.html}
}
@inproceedings{pinto2018asymmetric,
  title     = {Asymmetric Actor Critic for Image-Based Robot Learning},
  author    = {Pinto, Lerrel and Andrychowicz, Marcin and Welinder, Peter and Zaremba, Wojciech and Abbeel, Pieter},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year      = {2018},
  doi       = {10.15607/RSS.2018.XIV.008},
  eprint    = {1710.06542}, archivePrefix = {arXiv},
  url       = {https://www.roboticsproceedings.org/rss14/p08.html}
}

% ---------- Auxiliary tasks, maps, spatial memory ----------
@inproceedings{jaderberg2017unreal,
  title     = {Reinforcement Learning with Unsupervised Auxiliary Tasks},
  author    = {Jaderberg, Max and Mnih, Volodymyr and Czarnecki, Wojciech Marian and Schaul, Tom and Leibo, Joel Z. and Silver, David and Kavukcuoglu, Koray},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
  eprint    = {1611.05397}, archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1611.05397}
}
@inproceedings{lample2017fps,
  title     = {Playing {FPS} Games with Deep Reinforcement Learning},
  author    = {Lample, Guillaume and Chaplot, Devendra Singh},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {31}, number = {1},
  year      = {2017},
  doi       = {10.1609/aaai.v31i1.10827},
  eprint    = {1609.05521}, archivePrefix = {arXiv},
  url       = {https://aaai.org/papers/10827-aaai-31-2017/}
}
@inproceedings{mirowski2017navigate,
  title     = {Learning to Navigate in Complex Environments},
  author    = {Mirowski, Piotr and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
  eprint    = {1611.03673}, archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1611.03673}
}
@inproceedings{gupta2017cmp,
  title     = {Cognitive Mapping and Planning for Visual Navigation},
  author    = {Gupta, Saurabh and Davidson, James and Levine, Sergey and Sukthankar, Rahul and Malik, Jitendra},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2616--2625},
  year      = {2017},
  eprint    = {1702.03920}, archivePrefix = {arXiv},
  url       = {https://openaccess.thecvf.com/content_cvpr_2017/html/Gupta_Cognitive_Mapping_and_CVPR_2017_paper.html}
}
@inproceedings{parisotto2018neuralmap,
  title     = {Neural Map: Structured Memory for Deep Reinforcement Learning},
  author    = {Parisotto, Emilio and Salakhutdinov, Ruslan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  eprint    = {1702.08360}, archivePrefix = {arXiv},
  url       = {https://iclr.cc/virtual/2018/poster/196}
}
@inproceedings{dosovitskiy2017dfp,
  title     = {Learning to Act by Predicting the Future},
  author    = {Dosovitskiy, Alexey and Koltun, Vladlen},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
  eprint    = {1611.01779}, archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1611.01779}
}

% ---------- 28: demonstrations + learned world models ----------
@inproceedings{hansen2023modem,
  title     = {{MoDem}: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations},
  author    = {Hansen, Nicklas and Lin, Yixin and Su, Hao and Wang, Xiaolong and Kumar, Vikash and Rajeswaran, Aravind},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  eprint    = {2212.05698}, archivePrefix = {arXiv},
  url       = {https://iclr.cc/virtual/2023/poster/11332}
}
@article{staley2024wmhd,
  title   = {Agent-Centric Human Demonstrations Train World Models},
  author  = {Staley, James and Short, Elaine and Goel, Shivam and Shukla, Yash},
  journal = {Reinforcement Learning Journal},
  volume  = {4}, pages = {1873--1886},
  year    = {2024},
  note    = {Presented at the Reinforcement Learning Conference (RLC) 2024},
  url     = {https://rlj.org/papers/staley2024agent}
}
@inproceedings{rafailov2021vmail,
  title     = {Visual Adversarial Imitation Learning using Variational Models},
  author    = {Rafailov, Rafael and Yu, Tianhe and Rajeswaran, Aravind and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year      = {2021},
  eprint    = {2107.08829}, archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2021/hash/1796a48fa1968edd5c5d10d42c7b1813-Abstract.html}
}
@misc{demoss2023ditto,
  title  = {{DITTO}: Offline Imitation Learning with World Models},
  author = {DeMoss, Branton and Duckworth, Paul and Foerster, Jakob and Hawes, Nick and Posner, Ingmar},
  year   = {2023},
  eprint = {2302.03086}, archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2302.03086}
}

% ---------- 29: sequence / event-prioritised replay ----------
@inproceedings{kauvar2023curious,
  title     = {Curious Replay for Model-based Adaptation},
  author    = {Kauvar, Isaac and Doyle, Chris and Zhou, Linqi and Haber, Nick},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series    = {PMLR}, volume = {202},
  year      = {2023},
  eprint    = {2306.15934}, archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v202/kauvar23a.html}
}
@inproceedings{kapturowski2019r2d2,
  title     = {Recurrent Experience Replay in Distributed Reinforcement Learning},
  author    = {Kapturowski, Steven and Ostrovski, Georg and Quan, John and Munos, R{\'e}mi and Dabney, Will},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  url       = {https://iclr.cc/virtual/2019/poster/648}
}
@misc{brittain2019pser,
  title  = {Prioritized Sequence Experience Replay},
  author = {Brittain, Marc and Bertram, Josh and Yang, Xuxi and Wei, Peng},
  year   = {2019},
  note   = {NeurIPS 2019 Deep Reinforcement Learning Workshop},
  eprint = {1905.12726}, archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/1905.12726}
}
```

## B) Verification table

| key | verified? | relevant contribution (one sentence, from the abstract) | could not confirm / notes |
|---|---|---|---|
| hafner2022crafter | yes | Introduces Crafter, an open-world survival game that scores agents on 22 semantically meaningful achievements within a 1M-step budget. | OpenReview 403; ICLR confirmed via arXiv comment, camera-ready header and iclr.cc. |
| matthews2024craftax | yes | Rewrites Crafter in JAX (Craftax-Classic, up to 250× faster) and extends it into the harder Craftax benchmark. | — |
| stanic2023crafterood | partially | Shows tuned PPO baselines do well on Crafter partly by reading the inventory display, and introduces CrafterOOD plus object-centric agents for out-of-distribution generalization. | Venue and year confirmed. DOI 10.1109/TG.2023.3276849 seen only via doi.org lookup because the IEEE page didn't render. Volume, issue and pages unconfirmed. |
| walker2023modelbased | yes | Finds model-based agents pre-trained with intrinsic exploration transfer better than model-free baselines, including on Crafter. | — |
| moon2023achievement | yes | Shows well-tuned PPO is strong on Crafter and adds a contrastive "achievement distillation" loss so the agent predicts its next achievement. | Your citation omits author Bumsoo Park. |
| hafner2022director | yes | Learns hierarchical behavior from pixels by having a manager choose latent goals inside a learned world model, which a worker then reaches. | NeurIPS volume number not seen, so omitted. |
| du2023ellm | yes | Rewards an RL agent for reaching goals an LLM suggests from a text caption of its state, evaluated on Crafter and Housekeep. | — |
| wu2023spring | yes | Prompts an LLM (GPT-4) with Crafter's LaTeX paper and a DAG of questions to choose actions with no training. | The proceedings title and author order (Min before Prabhumoye) differ from arXiv; BibTeX follows the proceedings. DOI seen on Curran's proceedings.com page. |
| pomerleau1988alvinn | yes | A 3-layer network trained on simulated road images outputs steering directions for road following, an early learned driving policy trained to imitate target steering. | NeurIPS lists it under 1988 (Advances in NIPS 1). I did not confirm the 1989 print date. Page range 305–313 comes from the ACM DL and CMU pages, consistent with the PDF's page numbers. |
| ross2011dagger | yes | Introduces DAgger, which iteratively collects states under the learned policy and labels them with the expert, with no-regret guarantees against the drift between training and test state distributions. | PMLR page numbers not seen. |
| hester2018dqfd | yes | Combines TD updates with a supervised large-margin loss on demonstrator actions, plus prioritized replay, to speed up DQN with small amounts of demonstration data. | 14 authors. |
| nair2018overcoming | partially | Adds a demonstration replay buffer, a behavior-cloning loss with a Q-filter, and resets to demonstration states on top of DDPG+HER to solve long-horizon sparse-reward robot tasks. | ICRA 2018 confirmed via arXiv comment and dblp. DOI 10.1109/ICRA.2018.8463162 and pages 6292–6299 seen only via doi.org/dblp. |
| rajeswaran2018dapg | yes | Shows a small number of human demonstrations (behavior-cloning initialization plus an augmented policy gradient) cuts sample complexity for 24-DoF dexterous manipulation. | — |
| baker2022vpt | yes | Labels large amounts of unlabeled Minecraft video with an inverse dynamics model, then trains a behavioral prior by imitation that can be fine-tuned with imitation learning or RL. | 9 authors. |
| guss2019minerl | yes | Releases more than 60M automatically annotated state-action pairs of human Minecraft demonstrations. | — |
| ecoffet2021firstreturn | yes | Go-Explore remembers promising states, returns to them, then explores, solving hard-exploration Atari games. | — |
| ecoffet2019goexplore | yes | Original Go-Explore preprint, which robustifies trajectories found by exploration with imitation learning. | arXiv only. |
| andrychowicz2017her | yes | Relabels failed episodes with goals that were actually achieved, so off-policy RL can learn from sparse binary rewards. | 10 authors. Page numbers seen only on the ACM DL, so omitted. |
| argall2009survey | partially | Surveys robot learning from demonstration and categorizes how demonstrations are gathered and how policies are derived from them. | Volume and pages taken from the publisher-typeset PDF on the CMU site. DOI 10.1016/j.robot.2008.10.024 not seen on a publisher page; issue 5 seen only on the ACM DL. |
| vapnik2009lupi | partially | Introduces Learning Using Privileged Information (LUPI): a teacher provides extra information x* at training time that is unavailable at test time. | Volume 22 and pages 544–557 taken from a mirrored typeset PDF. DOI 10.1016/j.neunet.2009.06.042 and issue 5–6 seen only via secondary sources. |
| chen2019lbc | yes | Trains a privileged agent that sees ground-truth maps, then uses it as a teacher for a vision-only driving agent. | PMLR BibTeX gives year 2020 for CoRL 2019. |
| pinto2018asymmetric | yes | Trains the critic on full simulator state while the actor sees only images. | — |
| jaderberg2017unreal | yes | Adds auxiliary pseudo-reward control and reward-prediction tasks to A3C. Reward prediction samples replay so rewarding and non-rewarding sequences are equally represented (also relevant to item 29). | ICLR confirmed via ML Anthology and researchr only, because OpenReview returned 403. |
| lample2017fps | yes | Jointly trains a DRQN on Doom with an auxiliary loss for predicting game features (for example, enemy presence) that are available only at training time. | — |
| mirowski2017navigate | yes | Adds auxiliary depth-prediction and loop-closure tasks to RL navigation in 3D mazes. | 12 authors. |
| gupta2017cmp | yes | Builds a top-down belief map from first-person views with a differentiable planner trained end to end. | arXiv v3 is the IJCV version with an extra author. DOI 10.1109/CVPR.2017.769 seen only via doi.org. |
| parisotto2018neuralmap | yes | Uses a spatially structured 2D memory with location-based writes for deep RL agents. | — |
| dosovitskiy2017dfp | yes | Trains sensorimotor control by supervised prediction of future measurements, with goals specified at test time. | — |
| hansen2023modem | yes | Uses a few demonstrations with model-based RL (TD-MPC, not Dreamer) through behavior-cloning pretraining, seeding and oversampling demonstrations. | Confirmed in the paper text that it builds on TD-MPC. |
| staley2024wmhd | yes | Injects human teleoperated demonstrations into DreamerV3 world-model training, not into the policy; the actor still learns in imagination. | The RLJ page orders authors Staley, Short, Goel, Shukla, while the PDF has Short last. I couldn't confirm which order is canonical. |
| rafailov2021vmail | yes | Trains a variational latent dynamics model and a discriminator for adversarial imitation, with policy rollouts inside the model. | Adversarial imitation, not a behavior-cloning loss. |
| demoss2023ditto | yes | Offline imitation via RL inside a learned world model, penalizing latent divergence from expert trajectories. | arXiv only; no venue found. |
| kauvar2023curious | yes | Prioritized replay for DreamerV3 using visit counts and world-model loss, improving the Crafter score to 19.4. | PMLR page numbers not seen. |
| kapturowski2019r2d2 | yes | Prioritizes replayed sequences by mixing the maximum and mean TD error, because averaging over long sequences "tends to wash out large errors". | No arXiv version found. |
| brittain2019pser | yes | Extends prioritized experience replay so that priority is spread backward to the transitions leading into important events. | NeurIPS 2019 Deep RL workshop only. |

On item 28, I found no paper that adds a behavior-cloning loss on scripted demonstrations to a Dreamer-style actor that learns in imagination. The candidates are near misses:

- **MoDem** uses behavior cloning plus a world model, but the world model is TD-MPC.
- **The human-demonstrations paper** (Staley et al., RLJ 2024) uses DreamerV3, but demonstrations only train the world model.
- **V-MAIL** and **DITTO** do imitation inside a learned latent model, with adversarial or latent-distance rewards rather than behavior cloning.

A CVPR 2026 workshop paper, "Imitation learning through imagination in latent space", also came up. I did not fully verify it, so it's excluded.

On item 29, I found no paper that specifically shows rare key actions in demonstrations being under-weighted by uniform window sampling. The closest mechanisms are:

- **R2D2**: the max-plus-mean sequence priority.
- **PSER**: decays priority backward to the transitions before an important event.
- **Curious Replay**: prioritized replay for Dreamer.
- **UNREAL**: rebalanced sampling of rewarding events for its reward-prediction head.
- **MoDem**: oversamples demonstration data.

## C) arXiv policy, fetched today

**(a) Endorsement.** Since 21 January 2026, an institutional email alone no longer gets new submitters endorsed, in any category. You are endorsed automatically only if you have both an institutional email and claimed authorship of an existing paper in the same endorsement domain. Otherwise you need a personal endorsement from an established arXiv author in that domain. cs.AI and cs.LG share the cs endorsement domain, since most high-level subject areas are domains. You need at least one positive endorsement per category, and arXiv staff cannot endorse you or waive the requirement. Endorsers check that the paper is appropriate for the area; endorsement is not peer review.
- https://info.arxiv.org/help/endorsement.html
- https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/

**(b) Moderation.** Moderation is not peer review. Moderators check:
- **Scholarly standards:** carefully prepared sections, figures, tables and references, in a professional, neutral tone.
- **Scholarly interest:** originality and significance, with no plagiarism or misrepresentation. Course projects, research proposals and papers needing major revision can be declined.
- **Category fit:** moderators can reclassify papers or remove cross-lists.
- **Duplicates, rights and offensive images.**
- **Generative AI:** significant use of text-generating AI must be reported, and AI tools cannot be listed as authors.
- **Rate limit:** at most 2 new submissions per author per calendar month and 3 active at once.

Separately, the CS section now requires review/survey and position papers to have passed peer review at a journal or conference first. This does not affect ordinary research papers.
- https://info.arxiv.org/help/moderation/index.html
- https://blog.arxiv.org/2025/10/31/attention-authors-updated-practice-for-review-articles-and-position-papers-in-arxiv-cs-category/

**(c) Formats.**
- **Text, in order of preference:** (La)TeX, AMS(La)TeX or PDFLaTeX; then PDF; then HTML with JPEG/PNG/GIF images. arXiv rejects DVI, PS, PDF produced from TeX source, and scanned documents.
- **Figures with pdflatex ("LaTeX in PDF mode"):** PDF, PNG or JPG. Plain LaTeX in DVI mode accepts only PS/EPS. Don't mix the two kinds; arXiv does no on-the-fly conversion such as `-eps-converted-to.pdf`.
- **Other rules:** use `graphicx`; don't embed JavaScript or animations in PDFs; file names are case-sensitive and limited to `a-z A-Z 0-9 _ + - . , =`. Processing uses TeX Live 2025 by default (2023 is also available).
- https://info.arxiv.org/help/submit/index.html
- https://info.arxiv.org/help/submit_tex.html

**(d) `.bbl` files.** arXiv now runs `bibtex` itself, or `biber` when it detects biblatex. Either upload the `.bib` files or a pre-built `.bbl`. If present, the `.bbl` is used, and its name must match the main `.tex` file. If there's no `.bbl` and a needed `.bib` is missing, the submission is blocked. Other processors such as `bibtex8` or `upbibtex` require uploading a pre-built `.bbl`. A biblatex `.bbl` must come from the same backend as the document and use a compatible format: version 3.3 for TeX Live 2025, or 3.2/3.3 for TeX Live 2023. Other auxiliary files are deleted, except `.bbl` and `.ind`. Source: https://info.arxiv.org/help/submit_tex.html

I'm about to write the final answer, and I need to make sure it includes a table, since the user explicitly asked for one.

