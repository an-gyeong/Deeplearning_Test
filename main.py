from RNN_BrentOilPrices import loader
from RNN_BrentOilPrices import function
from RNN_BrentOilPrices import model

path = "D:/0403_0408/content/data/BrentOilPrices.csv"

def main():
    df = loader(path)
    function.trans_date(df, 'Date', 'Date') #str데이터 datetime형으로 변환
    x = function.MinMaxScaling(df, 'Price') #타겟데이터를 스케일링하고 리스트 형테로 변환 
    #print(function.validation_split(x, 800, 1000))
    #print(function.trans_date(df, 'Date', 'Date'))
    data_set = function.validation_split(x, 800, 1000)
    data_result = function.data_reshape(data_set[0],data_set[1])
    model.model_fit(data_result[0],data_result[1],data_result[2],data_result[3])
if __name__ == '__main__':
    main()