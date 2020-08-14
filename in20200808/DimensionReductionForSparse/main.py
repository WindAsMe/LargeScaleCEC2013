from in20200808.DimensionReductionForSparse.Sparse import SparseModel
from in20200808.DimensionReductionForSparse.util import benchmark, help, aim
from in20200808.DimensionReductionForSparse.DE import DE
from sklearn.preprocessing import PolynomialFeatures
from in20200710.BatchSparseTrainOptimization.util import help as new_help
import numpy as np
import time
from scipy.stats import mannwhitneyu


if __name__ == '__main__':
    Dim = 1000
    feature_size = 100000
    degree = 2
    func_num = 3
    benchmark_function = benchmark.f_evaluation
    benchmark_summary = benchmark.f_summary(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    max_variables_num = 50
    mini_batch_size = 1000
    evaluate_function = aim.fitness_evaluation
    name = 'f' + str(func_num)
    print(name, 'Optimization')
    # time_Lasso_start = time.process_time()
    # reg_Lasso, feature_names = SparseModel.Regression(degree, feature_size, Dim, mini_batch_size, scale_range,
    #                                                   benchmark_function)
    # time_Lasso_end = time.process_time()
    #
    # coef, feature_names = help.not_zero_feature(reg_Lasso.coef_, help.feature_names_normalization(feature_names))
    # time_grouping_start = time.process_time()

    groups_Lasso = [[0, 128, 129, 130, 134, 7, 8, 135, 11, 139, 140, 14, 143, 148, 22, 23, 153, 28, 156, 158, 159, 32, 34, 37, 166, 167, 41, 46, 49, 58, 60, 62, 64, 67, 73, 76, 81, 82, 86, 92, 102, 104, 105, 106, 110, 113, 118, 119, 120, 123], [1, 3, 4, 132, 6, 133, 138, 19, 152, 155, 30, 31, 40, 172, 45, 177, 51, 180, 54, 183, 57, 185, 186, 68, 69, 196, 197, 203, 204, 77, 206, 79, 213, 215, 88, 217, 218, 220, 94, 95, 96, 98, 99, 100, 107, 111, 112, 114, 115, 122], [2, 12, 13, 141, 144, 145, 18, 150, 25, 29, 160, 163, 165, 169, 170, 43, 44, 48, 53, 181, 182, 56, 59, 188, 191, 192, 198, 199, 72, 202, 205, 78, 80, 83, 211, 219, 93, 221, 222, 230, 103, 232, 233, 234, 237, 117, 247, 248, 121, 253], [258, 255, 260, 5, 263, 264, 265, 266, 268, 16, 272, 274, 21, 277, 27, 157, 286, 287, 33, 290, 36, 39, 168, 296, 42, 171, 173, 175, 179, 52, 309, 312, 316, 63, 66, 194, 200, 87, 216, 89, 224, 101, 229, 235, 109, 241, 242, 250, 126, 127], [262, 136, 9, 271, 146, 20, 151, 280, 281, 26, 284, 289, 292, 301, 302, 303, 304, 308, 310, 55, 184, 311, 315, 193, 322, 70, 71, 326, 330, 75, 331, 208, 209, 336, 85, 342, 344, 90, 348, 349, 350, 97, 226, 231, 108, 238, 240, 246, 251, 125], [256, 382, 386, 387, 261, 388, 389, 392, 137, 10, 267, 395, 397, 270, 398, 17, 402, 276, 278, 285, 162, 295, 305, 187, 61, 318, 195, 328, 201, 333, 334, 207, 335, 337, 210, 84, 223, 353, 227, 358, 363, 365, 366, 369, 371, 373, 375, 249, 379, 254], [385, 131, 393, 15, 401, 403, 404, 410, 415, 288, 416, 417, 421, 294, 422, 423, 297, 425, 426, 428, 176, 307, 437, 439, 441, 314, 442, 444, 445, 319, 320, 323, 324, 454, 456, 457, 74, 332, 465, 470, 343, 347, 482, 228, 356, 357, 370, 243, 245, 381], [384, 513, 514, 518, 521, 273, 149, 405, 407, 24, 409, 154, 411, 538, 539, 541, 545, 418, 291, 546, 550, 424, 298, 174, 178, 440, 189, 446, 325, 453, 458, 339, 469, 472, 475, 478, 225, 354, 355, 486, 361, 364, 494, 124, 497, 498, 244, 507, 380, 383], [517, 394, 396, 400, 147, 406, 279, 408, 282, 283, 412, 413, 414, 161, 35, 164, 293, 429, 431, 436, 448, 65, 452, 327, 461, 464, 338, 466, 212, 340, 467, 471, 352, 483, 484, 485, 359, 488, 362, 236, 493, 239, 368, 495, 496, 499, 501, 503, 506, 510], [636, 646, 650, 654, 527, 655, 658, 532, 661, 534, 664, 666, 668, 674, 420, 548, 38, 552, 430, 558, 432, 433, 565, 570, 443, 190, 578, 580, 586, 588, 468, 601, 346, 477, 351, 480, 481, 607, 608, 613, 617, 618, 619, 623, 626, 627, 374, 376, 252, 637], [512, 641, 516, 644, 645, 519, 522, 651, 524, 525, 142, 653, 275, 665, 669, 542, 543, 670, 549, 553, 299, 47, 559, 560, 306, 568, 571, 575, 579, 583, 589, 592, 593, 595, 214, 600, 602, 604, 612, 360, 377, 491, 620, 622, 372, 628, 505, 635, 509, 639], [640, 643, 648, 529, 657, 531, 660, 540, 671, 676, 677, 679, 682, 684, 685, 686, 688, 689, 50, 435, 690, 693, 697, 573, 702, 321, 705, 582, 710, 712, 585, 713, 459, 590, 591, 727, 728, 91, 731, 606, 736, 737, 487, 490, 492, 630, 631, 632, 508, 638], [642, 515, 647, 649, 779, 269, 781, 782, 528, 785, 787, 662, 790, 792, 799, 544, 673, 801, 803, 817, 434, 691, 821, 695, 826, 830, 581, 584, 715, 460, 716, 718, 721, 724, 473, 474, 603, 730, 610, 740, 744, 745, 746, 621, 752, 116, 756, 760, 763, 511], [257, 771, 903, 904, 908, 910, 784, 914, 915, 789, 536, 795, 923, 926, 675, 678, 551, 815, 816, 818, 692, 820, 567, 696, 827, 829, 449, 450, 451, 837, 455, 840, 717, 723, 852, 341, 853, 729, 865, 866, 739, 741, 869, 743, 489, 749, 879, 888, 634, 892], [259, 391, 775, 776, 780, 783, 530, 659, 537, 793, 667, 794, 672, 419, 807, 554, 562, 563, 819, 694, 825, 700, 828, 577, 706, 707, 709, 711, 841, 714, 587, 842, 463, 722, 725, 732, 735, 615, 747, 748, 750, 624, 753, 754, 755, 500, 502, 633, 762, 767], [897, 902, 905, 906, 907, 893, 535, 920, 924, 927, 928, 929, 804, 932, 935, 681, 811, 300, 944, 946, 947, 564, 951, 824, 954, 956, 317, 704, 834, 838, 845, 846, 594, 597, 726, 854, 345, 858, 859, 861, 862, 867, 876, 878, 881, 761, 890, 891, 765, 766], [896, 773, 520, 778, 909, 911, 788, 925, 798, 802, 936, 937, 938, 555, 809, 814, 943, 942, 822, 823, 313, 574, 831, 960, 833, 965, 839, 967, 969, 329, 971, 719, 976, 848, 856, 987, 860, 733, 990, 738, 870, 998, 367, 751, 629, 757, 759, 504, 758, 378], [768, 898, 900, 901, 390, 777, 523, 652, 399, 912, 786, 533, 919, 800, 934, 680, 808, 683, 556, 813, 941, 812, 561, 945, 949, 569, 699, 703, 959, 708, 836, 970, 972, 844, 462, 847, 720, 849, 978, 596, 855, 988, 863, 609, 996, 872, 873, 882, 885, 889], [769, 770, 774, 656, 917, 918, 791, 922, 797, 805, 810, 427, 939, 557, 438, 950, 952, 953, 698, 572, 447, 576, 961, 968, 973, 975, 977, 979, 851, 982, 983, 984, 985, 476, 989, 734, 991, 864, 993, 994, 995, 868, 479, 871, 616, 999, 625, 764, 894, 895], [899, 772, 526, 913, 916, 663, 921, 796, 930, 547, 931, 933, 806, 940, 687, 948, 566, 955, 701, 958, 957, 832, 962, 835, 963, 964, 966, 843, 974, 850, 980, 981, 598, 599, 857, 986, 605, 992, 611, 997, 614, 742, 874, 875, 877, 880, 883, 884, 886, 887]]
    # groups_random = help.groups_random_create(Dim, 25, 10)
    # groups_one = help.groups_one_create(Dim)
    # print('random grouping: ', groups_random)
    #
    # simple_problems_Dim, simple_problems_Data_index = help.extract(groups_modified)
    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the grouping strategy firstly do the best features combination in initial population
    simple_population_size = 30
    complex_population_size = 30000
    simple_MAX_iteration = 100
    complex_MAX_iteration = 100
    draw_simple_Max_iteration = 500
    draw_complex_Max_iteration = 2000

    test_times = 1
    efficient_Lasso_iteration_times = 0
    # efficient_random_iteration_times = 0
    # efficient_one_iteration_times = 0
    efficient_complex_iteration_times = 0

    max_or_min = 1

    # print(init_population)
    index = [0] * Dim
    best_simple = 0

    simple_Lasso_problems_trace = []
    # simple_random_problems_trace = []
    # simple_one_problems_trace = []
    complex_problems_trace = []

    best_Lasso_index_trace = []
    # best_random_index_trace = []
    # best_one_index_trace = []

    best_Lasso_index_average = []
    # best_random_index_average = []
    # best_one_index_average = []

    time_Lasso_group = 0
    # time_random_group = 0
    # time_one_group = 0
    time_normal = 0
    for t in range(test_times):
        print('round', t + 1)
        time1 = time.process_time()
        best_Lasso_obj_trace, best_Lasso_index, e_Lasso_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                                  simple_MAX_iteration,
                                                                                  benchmark_function, scale_range,
                                                                                  evaluate_function, groups_Lasso,
                                                                                  max_or_min)
        help.write_trace(name + '_LASSO', best_Lasso_obj_trace)
        time2 = time.process_time()
        efficient_Lasso_iteration_times += e_Lasso_time

        best_Lasso_index_trace.append(best_Lasso_index)

        time_Lasso_group += time2 - time1

        # best_random_obj_trace, best_random_index, e_random_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
        #                                                                           simple_MAX_iteration,
        #                                                                           benchmark_function, scale_range,
        #                                                                           evaluate_function, groups_random,
        #                                                                           max_or_min)
        # help.write_trace(name + '_random', best_random_obj_trace)
        #
        # efficient_random_iteration_times += e_random_time
        # best_random_index_trace.append(best_random_index)
        # time_random_group += time3 - time2
        # time3 = time.process_time()
        # best_one_obj_trace, best_one_index, e_one_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
        #                                                                           simple_MAX_iteration,
        #                                                                           benchmark_function, scale_range,
        #                                                                           evaluate_function, groups_one,
        #                                                                           max_or_min)
        #
        # help.write_trace(name + '_one', simple_one_problems_trace, best_one_obj_trace)
        time4 = time.process_time()
        # efficient_one_iteration_times += e_one_time
        # best_one_index_trace.append(best_one_index)
        # time_one_group += time4 - time3

        best_complex_trace, e_complex_time = DE.ComplexProblemsOptimization(Dim, complex_population_size, complex_MAX_iteration,
                                                            evaluate_function, benchmark_function, scale_range,
                                                            max_or_min)

        help.write_trace(name + '_normal', best_complex_trace)
        time5 = time.process_time()
        efficient_complex_iteration_times += e_complex_time
        time_normal += time5 - time4

    print('--------------------------------------------------------------------')
    print('Average Lasso group time: ', time_Lasso_group / test_times)
    # print('Average random group time: ', time_random_group / test_times)
    # print('Average one group time: ', time_one_group / test_times)
    print('Average normal time: ', time_normal / test_times)
    print('')
    print('Average efficient Lasso group : ', efficient_Lasso_iteration_times / test_times)
    # print('Average efficient random group time: ', efficient_random_iteration_times / test_times)
    # print('Average efficient one group time: ', efficient_one_iteration_times / test_times)
    print('Average efficient normal time: ', efficient_complex_iteration_times / test_times)
    simple_Lasso_problems_trace = np.array(simple_Lasso_problems_trace)
    # simple_random_problems_trace = np.array(simple_random_problems_trace)
    # simple_one_problems_trace = np.array(simple_one_problems_trace)

    # x1 = np.linspace(complex_population_size, complex_population_size * (draw_simple_Max_iteration + 1),
    #                  draw_simple_Max_iteration, endpoint=False)
    # x2 = np.linspace(complex_population_size, complex_population_size * (draw_complex_Max_iteration + 1),
    #                  draw_complex_Max_iteration, endpoint=False)
    # help.draw_obj(x1, x2, simple_Lasso_problems_trace_average[0:draw_simple_Max_iteration]
    #               , complex_problems_trace_average[0:draw_complex_Max_iteration], name)

    x = np.linspace(1, Dim + 1, Dim, endpoint=False)
    help.draw_var(x, best_Lasso_index_average, index, name)
    statistic = [10, 50, 100, 200]
    for s in statistic:
        print('Mannwhit statistic in ', str(s * complex_population_size), 'th Evaluation times: ',
              mannwhitneyu(simple_Lasso_problems_trace[:, s-1], complex_problems_trace[:, s-1]))

