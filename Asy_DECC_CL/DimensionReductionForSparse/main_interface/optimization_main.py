from Asy_DECC_CL.DimensionReductionForSparse.main_interface import f
from Asy_DECC_CL.DimensionReductionForSparse.util import help
from Asy_DECC_CL.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCDE, Normal
from cec2013lsgo.cec2013 import Benchmark
import time


if __name__ == '__main__':

    Dim = 1000
    NIND = 30
    bench = Benchmark()
    EFs = 3000000
    for func_num in [10]:
        test_time = 1
        name = 'f' + str(func_num)
        benchmark_summary = bench.get_info(func_num)
        scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

        # base_time = time.time()
        # groups_One = CCDE(Dim)
        # O_group_time = time.time()
        #
        # groups_DECC_DG, DECC_DG_cost = DECC_DG(func_num)
        # DG_group_time = time.time()

        # help.write_EFS_cost(name, 'DECC_DG_EFS', str(DECC_DG_cost))
        # [[0, 1, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 48], [3], [10, 30], [11], [24], [27], [34], [43], [47], [49], [44], [50], [51], [52], [53], [54], [55], [56], [57], [58, 75], [59], [60], [98, 61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [77, 71], [72], [73], [74], [76], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [90], [91], [92], [93], [94], [95], [96], [97], [99], [89], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145], [146], [147], [148], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [159], [160], [161], [162], [163], [164], [165], [166], [167], [168], [169], [170], [171], [172], [173], [174], [175], [176], [177], [178], [179], [180], [181], [182], [183], [184], [185], [186], [187], [188], [189], [190], [191], [192], [193], [194], [195], [196], [197], [198], [199], [200], [201], [202], [203], [204], [205], [206], [207], [208], [209], [210], [211], [212], [213], [214], [215], [216], [217], [218], [219], [220], [221], [222], [223], [224], [225], [226], [227], [228], [229], [230], [231], [232], [233], [234], [235], [236], [237], [238], [239], [240], [241], [242], [243], [244], [245], [246], [247], [248], [249], [250], [251], [252], [253], [254], [255], [256], [257], [258], [259], [260], [261], [262], [263], [264], [265], [266], [267], [268], [269], [270], [271], [272], [273], [274], [275], [276], [277], [278], [279], [280], [281], [282], [283], [284], [285], [286], [287], [288], [289], [290], [291], [292], [293], [294], [295], [296], [297], [298], [299], [300], [301], [302], [303], [304], [305], [306], [307], [308], [309], [310], [311], [312], [313], [314], [315], [316], [317], [318], [319], [320], [321], [322], [323], [324], [325], [326], [327], [328], [329], [330], [331], [332], [333], [334], [335], [336], [337], [338], [339], [340], [341], [342], [343], [344], [345], [346], [347], [348], [349], [350], [351], [352], [354], [356], [357], [358], [359], [360], [361], [362], [363], [364], [365], [366], [367], [368], [369], [370], [371], [372], [373], [374], [375], [376], [377], [378], [379], [380], [381], [382], [383], [384], [385], [386], [387], [388], [389], [390], [391], [392], [393], [394], [395], [396], [397], [398], [399], [353, 355], [400], [401], [402], [403], [404], [405], [406], [407], [408], [409], [410], [411], [412], [413], [414], [415], [416], [417], [418], [419], [420], [421], [422], [423], [424], [425], [426], [427], [428], [429], [430], [431], [432], [433], [434], [435], [436], [437], [438], [439], [440], [441], [442], [443], [444], [445], [446], [447], [448], [449], [450], [451], [452], [453], [454], [455], [456], [457], [458], [459], [460], [461], [462], [463], [464], [465], [466], [467], [468], [470], [471], [472], [474], [475], [476], [477], [478], [479], [480], [481], [482], [483], [484], [485], [486], [487], [488], [489], [490], [491], [492], [493], [494], [495], [496], [497], [498], [499], [469, 473], [500], [501], [502], [503], [506], [507], [508], [509], [510], [511], [513], [514], [515], [516], [517], [518], [519], [520], [521], [522], [523], [524], [525], [526], [529], [530], [531], [533], [535], [536], [537], [538], [539], [540], [541], [543], [544], [545], [546], [547], [549], [504, 505, 512, 527, 528, 532, 534, 542, 548], [551], [552], [553], [555], [556], [557], [559], [560], [563], [564], [565], [566], [567], [568], [569], [570], [571], [575], [576], [577], [578], [579], [581], [582], [583], [585], [586], [588], [589], [590], [591], [592], [594], [596], [597], [550, 554, 558, 561, 562, 572, 573, 574, 580, 584, 587, 593, 595, 598, 599], [600], [602], [604], [606], [607], [608], [615], [616], [619], [620], [621], [623], [624], [625], [626], [628], [629], [630], [631], [633], [635], [636], [638], [639], [640], [641], [642], [643], [644], [645], [646], [647], [649], [601, 603, 605, 609, 610, 611, 612, 613, 614, 617, 618, 622, 627, 632, 634, 637, 648], [650], [651], [654], [655], [659], [660], [662], [664], [666], [667], [670], [671], [672], [673], [676], [677], [678], [680], [681], [682], [683], [684], [685], [688], [690], [692], [693], [694], [695], [696], [697], [652, 653, 656, 657, 658, 661, 663, 665, 668, 669, 674, 675, 679, 686, 687, 689, 691, 698, 699], [700], [701], [704], [706], [707], [709], [711], [712], [714], [716], [717], [718], [722], [723], [724], [726], [730], [731], [732], [733], [736], [738], [741], [742], [744], [702, 703, 705, 708, 710, 713, 715, 719, 720, 721, 725, 727, 728, 729, 734, 735, 737, 739, 740, 743, 745, 746, 747, 748, 749], [751], [756], [760], [761], [763], [768], [771], [772], [773], [776], [779], [781], [782], [784], [785], [787], [788], [790], [794], [795], [796], [797], [798], [750, 752, 753, 754, 755, 757, 758, 759, 762, 764, 765, 766, 767, 769, 770, 774, 775, 777, 778, 780, 783, 786, 789, 791, 792, 793, 799], [802], [803], [807], [809], [810], [811], [814], [815], [817], [820], [821], [822], [823], [826], [834], [835], [837], [839], [841], [842], [844], [845], [847], [800, 801, 804, 805, 806, 808, 812, 813, 816, 818, 819, 824, 825, 827, 828, 829, 830, 831, 832, 833, 836, 838, 840, 843, 846, 848, 849], [851], [852], [855], [857], [858], [861], [864], [866], [870], [871], [873], [875], [876], [881], [882], [884], [887], [888], [890], [895], [896], [898], [850, 853, 854, 856, 859, 860, 862, 863, 865, 867, 868, 869, 872, 874, 877, 878, 879, 880, 883, 885, 886, 889, 891, 892, 893, 894, 897, 899], [901], [902], [904], [908], [914], [916], [917], [918], [919], [920], [922], [924], [927], [931], [932], [940], [942], [943], [944], [945], [948], [900, 903, 905, 906, 907, 909, 910, 911, 912, 913, 915, 921, 923, 925, 926, 928, 929, 930, 933, 934, 935, 936, 937, 938, 939, 941, 946, 947, 949], [950], [952], [958], [959], [960], [961], [965], [968], [969], [972], [974], [976], [977], [978], [980], [981], [982], [984], [986], [987], [988], [989], [991], [992], [994], [995], [996], [997], [998], [951, 953, 954, 955, 956, 957, 962, 963, 964, 966, 967, 970, 971, 973, 975, 979, 983, 985, 990, 993, 999]]
        # [[550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149], [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199], [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249], [250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299], [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349], [350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399], [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449], [450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499], [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549], [650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699], [700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749], [750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799], [850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899], [900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949]]
        for i in range(test_time):

            # groups_Normal = Normal(Dim)

            # b_time = time.time()
            # groups_DECC_G = DECC_G(Dim, 10, 100)
            # G_group_time = time.time()
            #
            # groups_DECC_D = DECC_D(func_num, 10, 100)
            # D_group_time = time.time()
            #
            groups_LASSO, LASSO_cost = LASSOCC(func_num)
            print(groups_LASSO)
            #
            # L_group_time = time.time()
            # help.write_EFS_cost(name, 'LASSO_cost_EFS', str(LASSO_cost))
            #
            # N_Opt_time = time.time()

            # f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_One, 'One')

            # O_Opt_time = time.time()

            # f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_DECC_G, 'DECC_G')
            # G_Opt_time = time.time()
            #
            # f.CC_exe(Dim, func_num, NIND, int((EFs - 100000) / (NIND * Dim)) - 2, scale_range, groups_DECC_D, 'DECC_D')
            # D_Opt_time = time.time()

            # f.CC_exe(Dim, func_num, NIND, int((EFs - DECC_DG_cost) / (NIND * Dim)) - 2, scale_range, groups_DECC_DG, 'DECC_DG')
            # DG_Opt_time = time.time()
            #
            # f.CC_exe(Dim, func_num, NIND, int((EFs - LASSO_cost) / (NIND * Dim)), scale_range, groups_LASSO, 'DECC_L')
            # L_Opt_time = time.time()
            #
            # f.DECC_CL_exe(Dim, func_num, NIND, scale_range, groups_One, groups_LASSO, LASSO_cost, 'DECC_CL')
            # CL_Opt_time = time.time()

            # help.write_CPU_cost(name, 'One_CPU', str(O_Opt_time - N_Opt_time + O_group_time - base_time))
            # help.write_CPU_cost(name, 'DECC_G_CPU', str(G_Opt_time - O_Opt_time + G_group_time - b_time))
            # help.write_CPU_cost(name, 'DECC_D_CPU', str(D_Opt_time - G_Opt_time + D_group_time - G_group_time))
            # help.write_CPU_cost(name, 'DECC_DG_CPU', str(DG_Opt_time - D_Opt_time + DG_group_time - O_group_time))
            # help.write_CPU_cost(name, 'DECC_L_CPU', str(L_Opt_time - DG_Opt_time + L_group_time - D_group_time))
            # help.write_CPU_cost(name, 'DECC_CL_CPU', str(CL_Opt_time - L_Opt_time + L_group_time - D_group_time))

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)


