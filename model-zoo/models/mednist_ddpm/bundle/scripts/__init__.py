import torch
from functools import partial
from monai import transforms as mt
from monai.utils import first, ensure_tuple, ensure_tuple_size
from monai.engines import PrepareBatch, default_prepare_batch

from typing import Dict, Mapping, Optional, Union


good_names = [ 
    "v19_153", "v19_223", "v20_500", "v20_195", "v20_016", "v20_635", "v19_089", "v19_227", "v19_048", "v19_078", "v19_262", "v20_529", "v20_620", "v20_453", "v19_145", "v19_215", "v19_261",
    "v20_757", "v19_083", "v20_707", "v20_546", "v20_525", "v20_521", "v20_766", "v20_505", "v19_243", "v20_352", "v20_763", "v20_558", "v19_113", "v20_714", "v19_124", "v19_250", "v19_139", 
    "v20_648", "v19_018", "v20_646", "v19_007", "v20_613", "v20_709", "v19_209", "v19_267", "v20_719", "v20_602", "v19_090", "v19_214", "v19_131", "v20_545", "v19_277", "v19_068", "v19_022", 
    "v20_585", "v19_080", "v19_031", "v20_580", "v19_102", "v19_278", "v20_754", "v20_760", "v20_590", "v20_507", "v19_207", "v19_260", "v19_147", "v19_138", "v19_251", "v20_631", "v20_711",
    "v20_586", "v20_017", "v19_091", "v20_825", "v20_701", "v20_544", "v19_266", "v19_023", "v19_104", "v20_700", "v19_212", "v19_026", "v20_716", "v19_247", "v20_647", "v19_205", "v20_607",
    "v19_208", "v19_149", "v19_275", "v20_518", "v19_010", "v19_013", "v19_046", "v19_070", "v19_055", "v19_130", "v19_005", "v19_085", "v19_108", "v19_134", "v20_596", "v20_756", "v20_811",
    "v20_513", "v19_273", "v19_107", "v19_254", "v19_143", "v20_565", "v20_604", "v20_755", "v20_640", "v19_008", "v19_058", "v20_603", "v20_802", "v19_016", "v19_054", "v20_419", "v19_060",
    "v19_217", "v19_127", "v19_253", "v19_051", "v19_059", "v19_075", "v20_761", "v20_619", "v20_616", "v19_020", "v20_703", "v19_043", "v19_155", "v20_506", "v19_256", "v20_649", "v20_479",
    "v19_257", "v20_816", "v19_154", "v19_272", "v19_226", "v19_004", "v19_065", "v20_769", "v19_239", "v19_230", "v20_768", "v19_067", "v20_713", "v20_295", "v19_202", "v20_572", "v19_030", 
    "v20_810", "v20_594", "v19_024", "v19_111", "v19_076", "v19_073", "v20_534", "v19_290", "v20_824", "v19_252", "v20_108", "v20_614", "v20_559", "v20_573", "v19_279", "v19_141", "v19_100",
    "v19_119", "v20_767", "v20_569", "v20_556", "v19_265", "v19_258", "v19_033", "v19_095", "v19_050", "v19_011", "v20_584", "v19_064", "v19_072", "v19_081", "v20_715", "v20_536", "v20_216",
    "v19_047", "v19_232", "v19_152", "v20_627", "v20_144", "v19_014", "v20_600", "v20_510", "v19_012", "v20_753", "v20_815", "v19_201", "v19_009", "v20_805", "v20_623", "v20_090", "v19_241",
    "v19_242", "v20_532", "v19_271",
]

good_images=[
    'VerSe19/dataset-verse19validation/rawdata/sub-verse153/sub-verse153_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse408/sub-verse408_split-verse223_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse500/sub-verse500_dir-ax_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-gl195/sub-gl195_dir-ax_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-gl016/sub-gl016_dir-ax_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse635/sub-verse635_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse089/sub-verse089_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse410/sub-verse410_split-verse227_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse048/sub-verse048_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse078/sub-verse078_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse407/sub-verse407_split-verse262_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse529/sub-verse529_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse620/sub-verse620_dir-iso_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-gl453/sub-gl453_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse145/sub-verse145_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse407/sub-verse407_split-verse215_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse406/sub-verse406_split-verse261_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse757/sub-verse757_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse083/sub-verse083_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse707/sub-verse707_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse546/sub-verse546_dir-ax_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse525/sub-verse525_dir-sag_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse521/sub-verse521_dir-ax_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse766/sub-verse766_dir-ax_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse505/sub-verse505_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse415/sub-verse415_split-verse243_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-gl352/sub-gl352_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse763/sub-verse763_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse558/sub-verse558_dir-sag_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse113/sub-verse113_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse714/sub-verse714_dir-iso_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse124/sub-verse124_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse250/sub-verse250_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse139/sub-verse139_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse648/sub-verse648_dir-iso_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse018/sub-verse018_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse646/sub-verse646_dir-iso_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse007/sub-verse007_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse613/sub-verse613_dir-iso_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse709/sub-verse709_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse404/sub-verse404_split-verse209_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse410/sub-verse410_split-verse267_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse719/sub-verse719_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse602/sub-verse602_dir-iso_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse400/sub-verse400_split-verse090_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse406/sub-verse406_split-verse214_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse131/sub-verse131_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse545/sub-verse545_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse417/sub-verse417_split-verse277_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse068/sub-verse068_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse022/sub-verse022_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse585/sub-verse585_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse080/sub-verse080_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse031/sub-verse031_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse580/sub-verse580_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse102/sub-verse102_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse417/sub-verse417_split-verse278_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse754/sub-verse754_dir-sag_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse760/sub-verse760_dir-iso_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse590/sub-verse590_dir-iso_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse507/sub-verse507_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse207/sub-verse207_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse260/sub-verse260_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse147/sub-verse147_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse138/sub-verse138_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse402/sub-verse402_split-verse251_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse631/sub-verse631_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse711/sub-verse711_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse586/sub-verse586_dir-iso_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-gl017/sub-gl017_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse091/sub-verse091_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse825/sub-verse825_dir-ax_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse701/sub-verse701_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse544/sub-verse544_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse409/sub-verse409_split-verse266_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse023/sub-verse023_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse104/sub-verse104_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse700/sub-verse700_dir-sag_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse405/sub-verse405_split-verse212_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse026/sub-verse026_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse716/sub-verse716_dir-iso_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse416/sub-verse416_split-verse247_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse647/sub-verse647_dir-sag_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse205/sub-verse205_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse607/sub-verse607_dir-sag_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse403/sub-verse403_split-verse208_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse149/sub-verse149_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse415/sub-verse415_split-verse275_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse518/sub-verse518_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse010/sub-verse010_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse013/sub-verse013_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse046/sub-verse046_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse070/sub-verse070_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse055/sub-verse055_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse130/sub-verse130_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse005/sub-verse005_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse085/sub-verse085_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse108/sub-verse108_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse134/sub-verse134_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse596/sub-verse596_dir-ax_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse756/sub-verse756_dir-iso_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse811/sub-verse811_dir-ax_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse513/sub-verse513_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse414/sub-verse414_split-verse273_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse107/sub-verse107_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse254/sub-verse254_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse143/sub-verse143_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse565/sub-verse565_dir-ax_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse604/sub-verse604_dir-iso_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse755/sub-verse755_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse640/sub-verse640_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse008/sub-verse008_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse058/sub-verse058_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse603/sub-verse603_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse802/sub-verse802_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse016/sub-verse016_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse054/sub-verse054_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-gl419/sub-gl419_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse060/sub-verse060_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse217/sub-verse217_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse127/sub-verse127_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse401/sub-verse401_split-verse253_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse051/sub-verse051_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse059/sub-verse059_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse075/sub-verse075_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse761/sub-verse761_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse619/sub-verse619_dir-ax_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse616/sub-verse616_dir-iso_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse020/sub-verse020_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse703/sub-verse703_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse043/sub-verse043_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse400/sub-verse400_split-verse155_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse506/sub-verse506_dir-iso_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse404/sub-verse404_split-verse256_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse649/sub-verse649_dir-sag_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-gl479/sub-gl479_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse257/sub-verse257_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse816/sub-verse816_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse154/sub-verse154_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse413/sub-verse413_split-verse272_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse409/sub-verse409_split-verse226_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse004/sub-verse004_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse065/sub-verse065_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse769/sub-verse769_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse413/sub-verse413_split-verse239_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse230/sub-verse230_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-verse768/sub-verse768_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse067/sub-verse067_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse713/sub-verse713_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-gl295/sub-gl295_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse402/sub-verse402_split-verse202_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse572/sub-verse572_dir-sag_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse030/sub-verse030_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-verse810/sub-verse810_dir-iso_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse594/sub-verse594_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse024/sub-verse024_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse111/sub-verse111_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse076/sub-verse076_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse073/sub-verse073_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse534/sub-verse534_dir-iso_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse412/sub-verse412_split-verse290_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse824/sub-verse824_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse252/sub-verse252_ct.nii.gz', 'VerSe20/03_test/rawdata/sub-gl108/sub-gl108_dir-ax_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse614/sub-verse614_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse559/sub-verse559_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse573/sub-verse573_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse416/sub-verse416_split-verse279_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse141/sub-verse141_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse100/sub-verse100_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse119/sub-verse119_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse767/sub-verse767_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse569/sub-verse569_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse556/sub-verse556_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse408/sub-verse408_split-verse265_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse405/sub-verse405_split-verse258_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse033/sub-verse033_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse095/sub-verse095_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse050/sub-verse050_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse011/sub-verse011_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse584/sub-verse584_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse064/sub-verse064_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse072/sub-verse072_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse081/sub-verse081_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse715/sub-verse715_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse536/sub-verse536_dir-ax_ct.nii.gz',
    'VerSe20/03_test/rawdata/sub-gl216/sub-gl216_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19validation/rawdata/sub-verse047/sub-verse047_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse411/sub-verse411_split-verse232_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse152/sub-verse152_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse627/sub-verse627_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-gl144/sub-gl144_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse014/sub-verse014_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse600/sub-verse600_dir-ax_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-verse510/sub-verse510_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse012/sub-verse012_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse753/sub-verse753_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse815/sub-verse815_ct.nii.gz',
    'VerSe19/dataset-verse19training/rawdata/sub-verse401/sub-verse401_split-verse201_ct.nii.gz', 'VerSe19/dataset-verse19training/rawdata/sub-verse009/sub-verse009_ct.nii.gz',
    'VerSe20/02_validation/rawdata/sub-verse805/sub-verse805_ct.nii.gz', 'VerSe20/02_validation/rawdata/sub-verse623/sub-verse623_ct.nii.gz',
    'VerSe20/01_training/rawdata/sub-gl090/sub-gl090_dir-ax_ct.nii.gz', 'VerSe19/dataset-verse19test/rawdata/sub-verse414/sub-verse414_split-verse241_ct.nii.gz',
    'VerSe19/dataset-verse19validation/rawdata/sub-verse242/sub-verse242_ct.nii.gz', 'VerSe20/01_training/rawdata/sub-verse532/sub-verse532_dir-ax_ct.nii.gz',
    'VerSe19/dataset-verse19test/rawdata/sub-verse271/sub-verse271_ct.nii.gz', 
]


def respace(img, new_space=1.0,mode="trilinear"):
    spatial_shape = torch.tensor(img.shape[1:])
    ssize = len(spatial_shape)

    new_space = ensure_tuple(new_space)
    new_space = torch.tensor(ensure_tuple_size(new_space, ssize, first(new_space)))

    old_space = torch.tensor(img.meta["pixdim"][1:4])

    new_shape = old_space.div(new_space).mul_(spatial_shape).long()

    return torch.nn.functional.interpolate(img[None], new_shape.tolist(),mode=mode)[0]


class Respaced(mt.Lambdad):
    def __init__(self, keys, new_space):
        super().__init__(keys, func=self._respace)
        self.new_space = new_space

    def _respace(self, x):
        return respace(x, self.new_space)


class DiffusionPrepareBatch(PrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    return the image and noise field as the image/target pair plus the noise field the kwargs under the key "noise".
    This assumes the inferer being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, num_train_timesteps: int, condition_name: Optional[str] = None):
        self.condition_name = condition_name
        self.num_train_timesteps = num_train_timesteps

    def get_noise(self, images):
        """Returns the noise tensor for input tensor `images`, override this for different noise distributions."""
        return torch.randn_like(images)

    def get_timesteps(self, images):
        return torch.randint(0, self.num_train_timesteps, (images.shape[0],), device=images.device).long()

    def __call__(
        self,
        batchdata: Dict[str, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        **kwargs,
    ):
        images, _ = default_prepare_batch(batchdata, device, non_blocking, **kwargs)
        noise = self.get_noise(images).to(device, non_blocking=non_blocking, **kwargs)
        timesteps = self.get_timesteps(images).to(device, non_blocking=non_blocking, **kwargs)

        kwargs = {"noise": noise, "timesteps": timesteps}

        if self.condition_name is not None and isinstance(batchdata, Mapping):
            kwargs["conditioning"] = batchdata[self.condition_name].to(device, non_blocking=non_blocking, **kwargs)

        # return input, target, arguments, and keyword arguments where noise is the target and also a keyword value
        return images, noise, (), kwargs


def inv_metric_cmp_fn(current_metric: float, prev_best: float) -> bool:
    """
    The default function to compare metric values between current metric and previous best metric.
    Args:
        current_metric: metric value of current round computation.
        prev_best: the best metric value of previous rounds to compare with.
    """
    return current_metric < prev_best
