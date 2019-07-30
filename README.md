# Open Images

------

### CSV2JSON

- #### File Structure

```
csv2json_code
    │
    ├── CONFIG
    │ 
    ├── csv2json.py
    │
    └── transfer_csv2json.sh

mmdet_data    
    │ 
    ├── challenge2018
    │
    ├── train08
    │
    └── validation
 
open_data    
    │ 
    ├── challenge-2019-label500-hierarchy.json
    │
    ├── challenge-2019-classes-description-500.csv
    │  
    ├── challenge-2019-train-detection-bbox.csv
    │
    └── challenge-2019-validation-detection-bbox.csv
```

- #### Running Models

```
sh csv2json_code/transfer_csv2json.sh
```

- ### Note!!!

  - 更改`CONFIG`中的路径
  - 数据集可以是zip文件，但代码会自动解压，注意留出足够空间
  - `train`,`val`,`test`都在`transfer_csv2json.sh`中，默认情况全运行，根据需要更改



------

### JSON2CSV

- #### Running Models

```
python json2csv_code/json2csv.py
```

- #### Note!!!

  更改`json2csv.py`中的三个路径，分别是`mmdet`测试之后的`json`文件，`CSV2JSON`之后的测试集的`json`文件，及你想要保存的`csv`文件地址

