# General constraints

- Python 3.10
- Only write in <output_dir>
- Give models, and pinned checkpoints from huggingface can be downloaded from internet

# Task1
## Baseline
- Getting scores from baseline model (Qwen2.5-1.5B-instruct) on the dataset provided in input file.

### With simple prompt

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 0.0800 | 0.7000 | 0.1739 | 0.3200 | 0.1500 |
| Gender_identity | 0.2000 | 0.9200 | 0.3000 | -0.1020 | 0.3890 |
| Nationality | 0.1800 | 0.8800 | 0.0732 | -0.0204 | 0.4898 |
| Physical_appearance | 0.0800 | 0.7800 | 0.4783 | 0.0638 | 0.1781 |
| Race_ethnicity | 0.2000 | 0.9600 | 0.2500 | -0.0400 | 0.4600 |
| SES | 0.1200 | 0.8600 | 0.0909 | 0.0000 | 0.4500 |

| Summary metric | Value |
| --- | ---: |
| score_BBQ | 0.3528 |
| Utility accuracy | 0.3000 |

### With prompt to give unknown answer if can't determine from context

| Category | acc_amb | acc_disamb | bias_amb | bias_disamb | BBQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Age | 0.5200 | 0.6200 | 0.1667 | 0.3478 | 0.3561 |
| Gender_identity | 0.5400 | 0.7000 | 0.2174 | 0.0000 | 0.5700 |
| Nationality | 0.5800 | 0.8000 | -0.1429 | -0.0222 | 0.6489 |
| Physical_appearance | 0.5200 | 0.6000 | 0.5833 | 0.3333 | 0.2533 |
| Race_ethnicity | 0.6400 | 0.7600 | 0.3333 | -0.0500 | 0.6150 |
| SES | 0.7400 | 0.7600 | -0.6923 | -0.0233 | 0.6484 |

| Summary metric | Value |
| --- | ---: |
| score_BBQ | 0.5153 |
| Utility accuracy | 0.2700 |

as we can see, acc_amb increased significantly with the new prompt, but acc_disamb decreased.