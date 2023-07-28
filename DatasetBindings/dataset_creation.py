import utils

path = "mls_italian_opus"
train = "train"
processed_info_1 = "PROCESSED/train.csv"

dict = utils.bind_audio_transcripts(path, train)
utils.save_dict_to_csv(dict, processed_info_1)

test = "test"
processed_info_2 = "PROCESSED/test.csv"

dict = utils.bind_audio_transcripts(path, test)
utils.save_dict_to_csv(dict, processed_info_2)

val = "dev"
processed_info_3 = "PROCESSED/validation.csv"

dict = utils.bind_audio_transcripts(path, val)
utils.save_dict_to_csv(dict, processed_info_3)