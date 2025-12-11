This repo contains the data and scaling law fitting code used in [Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training](https://arxiv.org/abs/2512.08894).

## Data

The data is stored in 2 csv files.

- `metrics_dclm_data_mixture.csv` contains architecture details, eval loss and benchmark accuracy results of models trained on the DCLM-based mixture, described in Section 3 of the paper.
- `metrics_c4_mixture.csv` contains architecture details, eval loss and benchmark accuracy results of models trained on the C4 dataset.

## Scaling Law Forms

In the directory `scaling_law_forms` we provide scripts for fitting scaling law forms analyzed in the paper.

- `equation_1_bnsl.py` contains fitting of Equation 1 (Section 3.2 of the paper).
- `equation_2_power_law.py` contains fitting of Equation 2 (Section 3.2 of the paper).
- `equation_4_multi_token_to_param_ratio.py` contains fitting of Equation 4 (Section 3.3 of the paper).
- `equation_5_pass_at_k.py` contains fitting of Equation 5 (Section 3.4 of the paper).
- `twostage_linear.py` contains fitting of the two stage approach with linear dependence of accuracy and the validation loss.
- `twostage_logistic.py` contains fitting of the two stage approach with dependence of accuracy from the validation loss described as logistic function.
- `equation_6_with_q_max` contains fitting of Equation 10 (Appendix L of the paper).

## Citation

If you find this work useful in your research, please cite:
```
@article{krajewski2025revisiting,
  title   = {Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training},
  author  = {Jakub Krajewski and Amitis Shidani and Dan Busbridge and Sam Wiseman and Jason Ramapuram},
  journal = {arXiv preprint arXiv:2512.08894},
  year    = {2025},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
