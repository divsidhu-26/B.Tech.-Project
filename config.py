# ----------------------- PATH ------------------------

ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
WN18_DATA_PATH = "%s/wn18" % DATA_PATH
WN18RR_DATA_PATH = "%s/wn18rr" % DATA_PATH
FB15K_DATA_PATH = "%s/fb15k" % DATA_PATH
FB15K237_DATA_PATH = "%s/fb15k-237" % DATA_PATH
FB3M_DATA_PATH = "%s/fb1m" % DATA_PATH
SOCCER_DATA_PATH = "%s/soccer" % DATA_PATH
SQA_DATA_PATH = "%s/sqa" % DATA_PATH
DATA_25_DATA_PATH = "%s/data_25" % DATA_PATH

LOG_PATH = "%s/log" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH
EMBEDDING_PATH = "~/Coding/NLP/WordEmbeddings"
# EMBEDDING_PATH = "/local/data2/pxu4/WordEmbeddings"

# ----------------------- DATA ------------------------

DATASET = {}

WN18_TRAIN_RAW = "%s/train.txt" % WN18_DATA_PATH
WN18_VALID_RAW = "%s/valid.txt" % WN18_DATA_PATH
WN18_TEST_RAW = "%s/test.txt" % WN18_DATA_PATH
WN18_TRAIN = "%s/digitized_train.txt" % WN18_DATA_PATH
WN18_VALID = "%s/digitized_valid.txt" % WN18_DATA_PATH
WN18_TEST = "%s/digitized_test.txt" % WN18_DATA_PATH
WN18_E2ID = "%s/e2id.txt" % WN18_DATA_PATH
WN18_R2ID = "%s/r2id.txt" % WN18_DATA_PATH

DATASET["wn18"] = {
    "train_raw": WN18_TRAIN_RAW,
    "valid_raw": WN18_VALID_RAW,
    "test_raw": WN18_TEST_RAW,
    "train": WN18_TRAIN,
    "valid": WN18_VALID,
    "test": WN18_TEST,
    "e2id": WN18_E2ID,
    "r2id": WN18_R2ID,
}

WN18RR_TRAIN_RAW = "%s/train.txt" % WN18RR_DATA_PATH
WN18RR_VALID_RAW = "%s/valid.txt" % WN18RR_DATA_PATH
WN18RR_TEST_RAW = "%s/test.txt" % WN18RR_DATA_PATH
WN18RR_TRAIN = "%s/digitized_train.txt" % WN18RR_DATA_PATH
WN18RR_VALID = "%s/digitized_valid.txt" % WN18RR_DATA_PATH
WN18RR_TEST = "%s/digitized_test.txt" % WN18RR_DATA_PATH
WN18RR_E2ID = "%s/e2id.txt" % WN18RR_DATA_PATH
WN18RR_R2ID = "%s/r2id.txt" % WN18RR_DATA_PATH

DATASET["wn18rr"] = {
    "train_raw": WN18RR_TRAIN_RAW,
    "valid_raw": WN18RR_VALID_RAW,
    "test_raw": WN18RR_TEST_RAW,
    "train": WN18RR_TRAIN,
    "valid": WN18RR_VALID,
    "test": WN18RR_TEST,
    "e2id": WN18RR_E2ID,
    "r2id": WN18RR_R2ID,
}

DATA_25_TRAIN_RAW = "%s/train.txt" % DATA_25_DATA_PATH
DATA_25_VALID_RAW = "%s/valid.txt" % DATA_25_DATA_PATH
DATA_25_TEST_RAW = "%s/test.txt" % DATA_25_DATA_PATH
DATA_25_TRAIN = "%s/digitized_train.txt" % DATA_25_DATA_PATH
DATA_25_VALID = "%s/digitized_valid.txt" % DATA_25_DATA_PATH
DATA_25_TEST = "%s/digitized_test.txt" % DATA_25_DATA_PATH
DATA_25_E2ID = "%s/e2id.txt" % DATA_25_DATA_PATH
DATA_25_R2ID = "%s/r2id.txt" % DATA_25_DATA_PATH

DATASET["data_25"] = {
    "train_raw": DATA_25_TRAIN_RAW,
    "valid_raw": DATA_25_VALID_RAW,
    "test_raw": DATA_25_TEST_RAW,
    "train": DATA_25_TRAIN,
    "valid": DATA_25_VALID,
    "test": DATA_25_TEST,
    "e2id": DATA_25_E2ID,
    "r2id": DATA_25_R2ID,
}

FB15K_TRAIN_RAW = "%s/train.txt" % FB15K_DATA_PATH
FB15K_VALID_RAW = "%s/valid.txt" % FB15K_DATA_PATH
FB15K_TEST_RAW = "%s/test.txt" % FB15K_DATA_PATH
FB15K_TRAIN = "%s/digitized_train.txt" % FB15K_DATA_PATH
FB15K_VALID = "%s/digitized_valid.txt" % FB15K_DATA_PATH
FB15K_TEST = "%s/digitized_test.txt" % FB15K_DATA_PATH
FB15K_E2ID = "%s/e2id.txt" % FB15K_DATA_PATH
FB15K_R2ID = "%s/r2id.txt" % FB15K_DATA_PATH

DATASET["fb15k"] = {
    "train_raw": FB15K_TRAIN_RAW,
    "valid_raw": FB15K_VALID_RAW,
    "test_raw": FB15K_TEST_RAW,
    "train": FB15K_TRAIN,
    "valid": FB15K_VALID,
    "test": FB15K_TEST,
    "e2id": FB15K_E2ID,
    "r2id": FB15K_R2ID,
}

FB15K237_TRAIN_RAW = "%s/train.txt" % FB15K237_DATA_PATH
FB15K237_VALID_RAW = "%s/valid.txt" % FB15K237_DATA_PATH
FB15K237_TEST_RAW = "%s/test.txt" % FB15K237_DATA_PATH
FB15K237_TRAIN = "%s/digitized_train.txt" % FB15K237_DATA_PATH
FB15K237_VALID = "%s/digitized_valid.txt" % FB15K237_DATA_PATH
FB15K237_TEST = "%s/digitized_test.txt" % FB15K237_DATA_PATH
FB15K237_E2ID = "%s/e2id.txt" % FB15K237_DATA_PATH
FB15K237_R2ID = "%s/r2id.txt" % FB15K237_DATA_PATH

DATASET["fb15k237"] = {
    "train_raw": FB15K237_TRAIN_RAW,
    "valid_raw": FB15K237_VALID_RAW,
    "test_raw": FB15K237_TEST_RAW,
    "train": FB15K237_TRAIN,
    "valid": FB15K237_VALID,
    "test": FB15K237_TEST,
    "e2id": FB15K237_E2ID,
    "r2id": FB15K237_R2ID,
}

FB3M_TRAIN_RAW = "%s/train.txt" % FB3M_DATA_PATH
FB3M_VALID_RAW = "%s/valid.txt" % FB3M_DATA_PATH
FB3M_TEST_RAW = "%s/test.txt" % FB3M_DATA_PATH
FB3M_TRAIN = "%s/digitized_train.txt" % FB3M_DATA_PATH
FB3M_VALID = "%s/digitized_valid.txt" % FB3M_DATA_PATH
FB3M_TEST = "%s/digitized_test.txt" % FB3M_DATA_PATH
FB3M_E2ID = "%s/e2id.txt" % FB3M_DATA_PATH
FB3M_R2ID = "%s/r2id.txt" % FB3M_DATA_PATH

DATASET["fb3m"] = {
    "train_raw": FB3M_TRAIN_RAW,
    "valid_raw": FB3M_VALID_RAW,
    "test_raw": FB3M_TEST_RAW,
    "train": FB3M_TRAIN,
    "valid": FB3M_VALID,
    "test": FB3M_TEST,
    "e2id": FB3M_E2ID,
    "r2id": FB3M_R2ID,
}

SOCCER_TRAIN_RAW = "%s/train.txt" % SOCCER_DATA_PATH
SOCCER_VALID_RAW = "%s/valid.txt" % SOCCER_DATA_PATH
SOCCER_TEST_RAW = "%s/test.txt" % SOCCER_DATA_PATH
SOCCER_TRAIN = "%s/digitized_train.txt" % SOCCER_DATA_PATH
SOCCER_VALID = "%s/digitized_valid.txt" % SOCCER_DATA_PATH
SOCCER_TEST = "%s/digitized_test.txt" % SOCCER_DATA_PATH
SOCCER_E2ID = "%s/e2id.txt" % SOCCER_DATA_PATH
SOCCER_R2ID = "%s/r2id.txt" % SOCCER_DATA_PATH

DATASET["soccer"] = {
    "train_raw": SOCCER_TRAIN_RAW,
    "valid_raw": SOCCER_VALID_RAW,
    "test_raw": SOCCER_TEST_RAW,
    "train": SOCCER_TRAIN,
    "valid": SOCCER_VALID,
    "test": SOCCER_TEST,
    "e2id": SOCCER_E2ID,
    "r2id": SOCCER_R2ID,
}

SQA_TRAIN_RAW = "%s/train.txt" % SQA_DATA_PATH
SQA_VALID_RAW = "%s/valid.txt" % SQA_DATA_PATH
SQA_TEST_RAW = "%s/test.txt" % SQA_DATA_PATH
SQA_TRAIN = "%s/digitized_train.txt" % SQA_DATA_PATH
SQA_VALID = "%s/digitized_valid.txt" % SQA_DATA_PATH
SQA_TEST = "%s/digitized_test.txt" % SQA_DATA_PATH
SQA_E2ID = "%s/e2id.txt" % SQA_DATA_PATH
SQA_R2ID = "%s/r2id.txt" % SQA_DATA_PATH

DATASET["sqa"] = {
    "train_raw": SQA_TRAIN_RAW,
    "valid_raw": SQA_VALID_RAW,
    "test_raw": SQA_TEST_RAW,
    "train": SQA_TRAIN,
    "valid": SQA_VALID,
    "test": SQA_TEST,
    "e2id": SQA_E2ID,
    "r2id": SQA_R2ID,
}

FB_EMBEDDING = "%s/freebase-vectors-skipgram1000.bin" % EMBEDDING_PATH

# ----------------------- PARAM -----------------------

RANDOM_SEED = None
