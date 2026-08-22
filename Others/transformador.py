# Bases selecionada: bbc_text_cls, IMDB_Dataset, SMSSpamCollection, Yelp, SST2, News_Category_Dataset_v3,
# cyberbullying_tweets, fake_news_train, humor_classification_dataset, Womens_Clothing_E-Commerce_Reviews

# Caminho para a base
path = "data_base/data_for_AI_adapted.csv"

# Bibliotecas importadas
import pandas as pd
import numpy as np
import random
from pathlib import Path

path_1 = Path(path)

path_ = path_1.parent / "new_data_bases"

# Vai depender da base de dados que você quer ler podendo mudar para parquet ou qualquer outro tipo
if path_1.suffix == ".csv":
    data = pd.read_csv(filepath_or_buffer=path)
elif path_1.suffix == ".parquet":
    data = pd.read_parquet(path=path)
    path_1 = path_1.with_suffix(".csv")

label = data["labels"]

to_encode = list(label.unique())
numbers_class = label.nunique()

label_list = list(label)

flag = False
if label.dtype != (int or float or complex or bool):
    flag = True

if flag == True:
    from sklearn.preprocessing import OrdinalEncoder
    
    encoder = OrdinalEncoder(categories = [to_encode])
    encoder.fit(data[['labels']])

porcentages_turn = [1,5,10,15,20]

for percentage_turn in porcentages_turn:

    df = pd.DataFrame()

    df = data[["labels","text"]]

    df_2 = data[["labels","text"]]
    # Independente
    # if flag == True: 
    #     for i in range(10):
    #         percentage = 1 - np.random.uniform(0.4,0.6)
    #         sample = label.sample(frac=percentage)
    #         sample = sample.apply(lambda x: np.random.randint(0,numbers_class))
    #         re_sample = pd.DataFrame(sample)
    #         decoded_values = encoder.inverse_transform(re_sample).flatten()
    #         column_name = f'Labels_{i}'
    #         df[column_name] = label
    #         df.loc[sample.index,column_name] = decoded_values
    # else:
    #     for i in range(10):
    #         percentage = 1 - np.random.uniform(0.4,0.6)
    #         sample = label.sample(frac=percentage)
    #         sample = sample.apply(lambda x: np.random.randint(0,numbers_class))
    #         column_name = f'Labels_{i}'
    #         df[column_name] = label
    #         df.loc[sample.index,column_name] = sample

    # indexes_1 = random.sample(population=range(0,len(data["labels"])),k=int((len(data["labels"])*percentage_turn)/100))

    # num_cols = df.shape[1] - 2

    # values_1 = data.loc[indexes_1, ["labels"]].values

    # df.iloc[indexes_1, 2:] = np.tile(values_1, (1, num_cols))

    # new_path = f"{path_1.stem}_independent_transformed_{percentage_turn}{path_1.suffix}"

    # new_path = str(path_ / new_path)

    # df.to_csv(path_or_buf=new_path,index = False)
    #print(new_path)
    
    # Dependente
    if flag == True:
        percentage = 1 - np.random.uniform(0.4,0.6)
        sample = label.sample(frac=percentage)
        sample = sample.apply(lambda x: np.random.randint(0,numbers_class))
        re_sample = pd.DataFrame(sample)
        decoded_values = encoder.inverse_transform(re_sample).flatten()
        column_name = f'Labels_{10}'
        df_2[column_name] = label
        df_2.loc[sample.index,column_name] = decoded_values

        for i in range(9):
            percentage = 1 - np.random.uniform(0.4,0.6)
            label_2 = df_2[f"Labels_{i+10}"]
            sample = label_2.sample(frac=percentage)
            sample = sample.apply(lambda x: np.random.randint(0,numbers_class))
            re_sample = pd.DataFrame(sample)
            decoded_values = encoder.inverse_transform(re_sample).flatten()
            column_name = f'Labels_{i+11}'
            df_2[column_name] = label_2
            df_2.loc[sample.index,column_name] = decoded_values
    else:
        percentage = 1 - np.random.uniform(0.4,0.6)
        sample = label.sample(frac=percentage)
        sample = sample.apply(lambda x: np.random.randint(0,numbers_class))
        column_name = f'Labels_{10}'
        df_2[column_name] = label
        df_2.loc[sample.index,column_name] = sample
        
        for i in range(10):
            percentage = 1 - np.random.uniform(0.4,0.6)
            label_2 = df_2[f"Labels_{i+10}"]
            sample = label.sample(frac=percentage)
            sample = sample.apply(lambda x: np.random.randint(0,numbers_class))
            column_name = f'Labels_{i+11}'
            df_2[column_name] = label_2
            df_2.loc[sample.index,column_name] = sample

    indexes_2 = random.sample(population=range(0, len(data["labels"])), k=int((len(data["labels"])*percentage_turn)/100))

    num_cols_2 = df_2.shape[1] - 2 
    values_2 = data.loc[indexes_2, ["labels"]].values 
    
    df_2.iloc[indexes_2, 2:] = np.tile(values_2, (1, num_cols_2))

    new_path_2 = f"{path_1.stem}_dependent_transformed_{percentage_turn}{path_1.suffix}"

    new_path_2 = str(path_ / new_path_2)

    df_2.to_csv(path_or_buf=new_path_2,index = False)

    #print(new_path_2)

#df.to_csv(path_or_buf=new_path,index = False)