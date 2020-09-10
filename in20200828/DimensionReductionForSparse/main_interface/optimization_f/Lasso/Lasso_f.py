from in20200828.DimensionReductionForSparse.util import help
from in20200828.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def f():
    Dim = 1000
    func_num = 11
    bench = Benchmark()
    benchmark_function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

    name = 'f' + str(func_num)
    print(name, 'Optimization')
    print('scale range: ', scale_range)
    # groups_Lasso = [[0, 1, 5, 133, 135, 21, 151, 24, 26, 30, 31, 159, 35, 163, 37, 164, 169, 42, 45, 174, 176, 50, 51, 178, 179, 54, 183, 185, 187, 64, 73, 203, 81, 85, 213, 219, 222, 223, 224, 227, 230, 106, 108, 109, 237, 239, 113, 241, 125, 127], [128, 2, 131, 258, 6, 8, 136, 267, 16, 17, 19, 148, 152, 28, 157, 160, 33, 34, 39, 41, 46, 56, 184, 59, 188, 189, 190, 191, 193, 66, 67, 196, 74, 202, 209, 215, 88, 91, 248, 99, 104, 235, 238, 242, 244, 117, 245, 120, 122, 124], [3, 132, 134, 137, 10, 140, 13, 141, 142, 144, 146, 20, 155, 156, 166, 168, 170, 171, 47, 175, 49, 52, 53, 180, 55, 61, 62, 65, 194, 195, 68, 69, 197, 200, 201, 205, 206, 208, 210, 211, 86, 90, 100, 101, 112, 114, 118, 119, 121, 123], [129, 257, 4, 138, 11, 270, 15, 275, 149, 23, 282, 285, 162, 165, 293, 296, 43, 44, 173, 299, 300, 301, 177, 306, 308, 309, 182, 311, 63, 320, 71, 327, 328, 329, 76, 77, 78, 332, 333, 336, 341, 87, 89, 93, 98, 226, 232, 233, 240, 252], [384, 386, 387, 7, 391, 265, 266, 12, 14, 143, 277, 406, 154, 289, 36, 292, 38, 295, 298, 310, 319, 323, 70, 198, 72, 199, 326, 331, 335, 80, 82, 343, 347, 220, 221, 351, 352, 356, 358, 103, 359, 361, 236, 365, 366, 115, 373, 250, 253, 383], [389, 518, 263, 9, 397, 398, 272, 400, 529, 403, 279, 153, 25, 284, 413, 414, 287, 416, 167, 425, 426, 172, 305, 433, 58, 442, 316, 446, 447, 321, 461, 462, 339, 340, 216, 344, 477, 225, 353, 228, 105, 489, 363, 505, 246, 247, 377, 506, 507, 380], [264, 393, 394, 271, 273, 18, 22, 408, 410, 27, 283, 29, 412, 415, 288, 417, 418, 421, 294, 297, 302, 303, 307, 57, 314, 315, 60, 324, 330, 204, 334, 337, 83, 218, 346, 92, 349, 94, 96, 354, 229, 102, 360, 107, 367, 369, 376, 379, 254, 255], [256, 390, 396, 399, 147, 405, 409, 537, 540, 158, 286, 32, 161, 549, 422, 551, 424, 552, 555, 428, 557, 430, 558, 48, 567, 317, 192, 449, 453, 458, 75, 464, 465, 466, 469, 217, 476, 95, 97, 481, 231, 487, 488, 234, 362, 493, 368, 497, 370, 126], [512, 388, 261, 392, 526, 401, 274, 411, 419, 423, 40, 553, 429, 304, 561, 181, 568, 441, 570, 571, 577, 322, 450, 456, 584, 463, 592, 593, 212, 84, 468, 599, 600, 473, 474, 475, 348, 601, 350, 478, 603, 604, 485, 491, 496, 243, 371, 374, 504, 509], [513, 519, 520, 522, 268, 402, 530, 276, 531, 532, 407, 536, 545, 290, 556, 431, 435, 565, 438, 312, 313, 572, 573, 576, 578, 580, 325, 582, 586, 589, 590, 79, 595, 596, 342, 472, 606, 479, 480, 609, 482, 616, 618, 364, 623, 498, 500, 628, 251, 508], [640, 644, 647, 528, 658, 660, 661, 150, 280, 541, 542, 673, 675, 420, 676, 680, 682, 683, 432, 560, 688, 694, 699, 701, 318, 704, 454, 712, 457, 713, 723, 598, 727, 731, 605, 607, 736, 355, 483, 611, 612, 631, 490, 492, 110, 116, 503, 635, 637, 638], [769, 259, 771, 650, 395, 523, 269, 651, 766, 145, 659, 404, 534, 662, 665, 539, 667, 674, 550, 681, 685, 687, 569, 700, 702, 575, 705, 581, 710, 714, 459, 719, 594, 470, 728, 345, 608, 741, 746, 749, 750, 111, 495, 751, 752, 753, 755, 758, 249, 510], [641, 130, 514, 770, 649, 139, 652, 527, 656, 533, 538, 666, 668, 669, 671, 548, 686, 559, 690, 563, 692, 693, 695, 440, 696, 186, 698, 444, 579, 452, 707, 708, 583, 718, 720, 724, 725, 729, 732, 734, 735, 737, 484, 357, 486, 629, 502, 630, 636, 382], [646, 781, 655, 657, 786, 278, 281, 793, 795, 800, 802, 808, 810, 812, 813, 814, 689, 697, 825, 827, 828, 830, 448, 451, 711, 840, 717, 845, 207, 721, 338, 850, 851, 855, 856, 857, 859, 862, 865, 870, 871, 872, 878, 879, 882, 499, 883, 501, 762, 767], [642, 260, 772, 773, 776, 654, 783, 785, 788, 789, 791, 664, 798, 546, 803, 805, 678, 807, 815, 816, 436, 439, 835, 839, 585, 841, 842, 843, 844, 846, 848, 849, 722, 467, 214, 726, 854, 730, 858, 863, 610, 614, 742, 617, 621, 625, 372, 756, 759, 639], [643, 645, 262, 774, 777, 779, 663, 792, 796, 670, 543, 544, 801, 291, 547, 677, 806, 809, 427, 562, 818, 820, 437, 821, 822, 443, 574, 832, 706, 834, 836, 455, 460, 591, 847, 852, 866, 739, 869, 873, 875, 748, 494, 622, 881, 884, 885, 375, 761, 763], [775, 648, 905, 521, 907, 653, 782, 787, 535, 920, 797, 799, 935, 943, 817, 946, 691, 949, 823, 952, 445, 833, 962, 838, 970, 716, 973, 597, 471, 987, 860, 864, 992, 738, 995, 868, 996, 998, 867, 744, 874, 876, 877, 760, 888, 889, 378, 891, 890, 895], [768, 385, 896, 899, 516, 900, 901, 902, 904, 906, 780, 908, 909, 910, 784, 912, 914, 916, 790, 919, 921, 794, 923, 927, 672, 928, 929, 931, 804, 811, 434, 564, 824, 826, 829, 831, 588, 602, 861, 743, 619, 620, 627, 757, 886, 632, 633, 892, 381, 894], [898, 515, 517, 903, 524, 525, 913, 918, 922, 924, 925, 933, 937, 554, 939, 941, 944, 945, 948, 954, 955, 956, 957, 703, 960, 961, 964, 965, 967, 969, 587, 974, 975, 977, 980, 853, 984, 985, 733, 989, 990, 993, 994, 740, 999, 747, 624, 887, 765, 511], [897, 778, 911, 915, 917, 926, 930, 932, 934, 679, 936, 938, 940, 684, 942, 819, 947, 566, 950, 951, 953, 958, 959, 963, 709, 966, 837, 968, 971, 715, 972, 976, 978, 979, 981, 982, 983, 986, 988, 991, 613, 997, 615, 745, 880, 626, 754, 634, 764, 893]]
    groups_Lasso = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145], [146], [147], [148], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [159], [160], [161], [162], [163], [164], [165], [166], [167], [168], [169], [170], [171], [172], [173], [174], [175], [176], [177], [178], [179], [180], [181], [182], [183], [184], [185], [186], [187], [188], [189], [190], [191], [192], [193], [194], [195], [196], [197], [198], [199], [200], [201], [202], [203], [204], [205], [206], [207], [208], [209], [210], [211], [212], [213], [214], [215], [216], [217], [218], [219], [220], [221], [222], [223], [224], [225], [226], [227], [228], [229], [230], [231], [232], [233], [234], [235], [236], [237], [238], [239], [240], [241], [242], [243], [244], [245], [246], [247], [248], [249], [250], [251], [252], [253], [254], [255], [256], [257], [258], [259], [260], [261], [262], [263], [264], [265], [266], [267], [268], [269], [270], [271], [272], [273], [274], [275], [276], [277], [278], [279], [280], [281], [282], [283], [284], [285], [286], [287], [288], [289], [290], [291], [292], [293], [294], [295], [296], [297], [298], [299], [300], [301], [302], [303], [304], [305], [306], [307], [308], [309], [310], [311], [312], [313], [314], [315], [316], [317], [318], [319], [320], [321], [322], [323], [324], [325], [326], [327], [328], [329], [330], [331], [332], [333], [334], [335], [336], [337], [338], [339], [340], [341], [342], [343], [344], [345], [346], [347], [348], [349], [350], [351], [352], [353], [354], [355], [356], [357], [358], [359], [360], [361], [362], [363], [364], [365], [366], [367], [368], [369], [370], [371], [372], [373], [374], [375], [376], [377], [378], [379], [380], [381], [382], [383], [384], [385], [386], [387], [388], [389], [390], [391], [392], [393], [394], [395], [396], [397], [398], [399], [400], [401], [402], [403], [404], [405], [406], [407], [408], [409], [410], [411], [412], [413], [414], [415], [416], [417], [418], [419], [420], [421], [422], [423], [424], [425], [426], [427], [428], [429], [430], [431], [432], [433], [434], [435], [436], [437], [438], [439], [440], [441], [442], [443], [444], [445], [446], [447], [448], [449], [450], [451], [452], [453], [454], [455], [456], [457], [458], [459], [460], [461], [462], [463], [464], [465], [466], [467], [468], [469], [470], [471], [472], [473], [474], [475], [476], [477], [478], [479], [480], [481], [482], [483], [484], [485], [486], [487], [488], [489], [490], [491], [492], [493], [494], [495], [496], [497], [498], [499], [500], [501], [502], [503], [504], [505], [506], [507], [508], [509], [510], [511], [512], [513], [514], [515], [516], [517], [518], [519], [520], [521], [522], [523], [524], [525], [526], [527], [528], [529], [530], [531], [532], [533], [534], [535], [536], [537], [538], [539], [723, 540], [541], [542], [543], [544], [545], [546], [547], [548], [549], [550], [551], [552], [553], [554], [555], [556], [557], [558], [559], [560], [561], [562], [563], [564], [565], [566], [567], [568], [569], [570], [571], [572], [573], [574], [575], [576], [577], [578], [579], [580], [581], [582], [583], [584], [585], [586], [587], [588], [589], [590], [591], [592], [593], [594], [595], [596], [597], [598], [599], [600], [601], [602], [603], [604], [605], [606], [607], [608], [609], [610], [611], [612], [613], [614], [615], [616], [617], [618], [619], [620], [621], [622], [623], [624], [625], [626], [627], [628], [629], [630], [631], [632], [633], [634], [635], [636], [637], [638], [639], [640], [641], [642], [643], [644], [645], [646], [647], [648], [649], [650], [651], [652], [653], [654], [655], [656], [657], [658], [659], [660], [661], [662], [663], [664], [665], [666], [667], [668], [669], [670], [671], [672], [673], [674], [675], [676], [677], [678], [679], [680], [681], [682], [683], [684], [685], [686], [687], [688], [689], [690], [691], [692], [693], [694], [695], [696], [697], [698], [699], [700], [701], [702], [703], [704], [705], [706], [707], [708], [709], [710], [711], [712], [713], [714], [715], [716], [717], [718], [719], [720], [721], [722], [724], [725], [726], [727], [728], [729], [730], [731], [732], [733], [734], [735], [736], [737], [738], [739], [740], [741], [742], [743], [744], [745], [746], [747], [748], [749], [750], [751], [752], [753], [754], [755], [756], [757], [758], [759], [760], [761], [762], [763], [764], [765], [766], [767], [768], [769], [770], [771], [772], [773], [774], [775], [776], [777], [778], [779], [780], [781], [782], [783], [784], [785], [786], [787], [788], [789], [790], [791], [792], [793], [794], [795], [796], [797], [798], [799], [800], [801], [802], [803], [804], [805], [806], [807], [808], [809], [810], [811], [812], [813], [814], [815], [816], [817], [818], [819], [820], [821], [822], [823], [824], [825], [826], [827], [828], [829], [830], [831], [832], [833], [834], [835], [836], [837], [838], [839], [840], [841], [842], [843], [844], [845], [846], [847], [848], [849], [850], [851], [852], [853], [854], [855], [856], [857], [858], [859], [860], [861], [862], [863], [864], [865], [866], [867], [868], [869], [870], [871], [872], [873], [874], [875], [876], [877], [878], [879], [880], [881], [882], [883], [884], [885], [886], [887], [888], [889], [890], [891], [892], [893], [894], [895], [896], [897], [898], [899], [900], [901], [902], [903], [904], [905], [906], [907], [908], [909], [910], [911], [912], [913], [914], [915], [916], [917], [918], [919], [920], [921], [922], [923], [924], [925], [926], [927], [928], [929], [930], [931], [932], [933], [934], [935], [936], [937], [938], [939], [940], [941], [942], [943], [944], [945], [946], [947], [948], [949], [950], [951], [952], [953], [954], [955], [956], [957], [958], [959], [960], [961], [962], [963], [964], [965], [966], [967], [968], [969], [970], [971], [972], [973], [974], [975], [976], [977], [978], [979], [980], [981], [982], [983], [984], [985], [986], [987], [988], [989], [990], [991], [992], [993], [994], [995], [996], [997], [998], [999]]
    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the grouping strategy firstly do the best features combination in initial population
    simple_population_size = 30
    simple_MAX_iteration = 99
    test_times = 1
    max_or_min = 1

    for t in range(test_times):
        print('round', t + 1)
        best_Lasso_obj_trace, best_Lasso_index = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                               simple_MAX_iteration, benchmark_function,
                                                                               scale_range, groups_Lasso, max_or_min)
        help.write_obj_trace(name, 'LASSO', best_Lasso_obj_trace)
        help.write_var_trace(name, 'LASSO', best_Lasso_index)
        x = np.linspace(0, 3000000, simple_MAX_iteration)
        help.draw_obj(x, best_Lasso_obj_trace, 'LASSO Grouping', 'temp')