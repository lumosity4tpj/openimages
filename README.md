# Open Images

------



### File Structure

------

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



### Running Models

------

```
cd csv2json_code
sh transfer_csv2json.sh
```



### Note!!!

------

- 更改`CONFIG`中的路径
- 数据集可以是zip文件，但代码会自动解压，注意留出足够空间
- `train`,`val`,`test`都在`transfer_csv2json.sh`中，默认情况全运行，根据需要更改

