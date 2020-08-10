from in20200808.DimensionReductionForSparse.Sparse import SparseModel
from in20200808.DimensionReductionForSparse.util import benchmark, help, aim
from in20200808.DimensionReductionForSparse.DE import DE
from sklearn.preprocessing import PolynomialFeatures
from in20200710.BatchSparseTrainOptimization.util import help as new_help
import numpy as np
import time
from scipy.stats import kruskal


if __name__ == '__main__':
    Dim = 1000
    feature_size = 100000
    degree = 2
    func_num = 1
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

    groups_Lasso = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145], [146], [147], [148], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [159], [160], [161], [162], [163], [164], [165], [166], [167], [168], [169], [170], [171], [172], [173], [174], [175], [176], [177], [178], [179], [180], [181], [182], [183], [184], [185], [186], [187], [188], [189], [190], [191], [192], [193], [194], [195], [196], [197], [198], [199], [200], [201], [202], [203], [204], [205], [206], [207], [208], [209], [210], [211], [212], [213], [214], [215], [216], [217], [218], [219], [220, 270], [221], [222], [223], [224], [225], [226], [227], [228], [229], [230], [231], [232], [233], [234], [235], [236], [237], [238], [239], [240], [241], [242], [243], [244], [245], [246], [247], [248], [249], [250], [251], [252], [253], [254], [255], [256], [257], [258], [259], [260], [261], [262], [263], [264], [265], [266], [267], [268], [269], [271], [272], [273], [274], [275], [276], [277], [278], [279], [280], [281], [282], [283], [284], [285], [286], [287], [288], [289], [290], [291], [292], [293], [294], [295], [296], [297], [298], [299], [300], [301], [302], [303], [304], [305], [306], [307], [308], [309], [310], [311], [312], [313], [314], [315], [316], [317], [318], [319], [320], [321], [322], [323], [324], [325], [326], [327], [328], [329], [330], [331], [332], [333], [334], [335], [336], [337], [338], [339], [340], [341], [342], [343], [344], [345], [346], [347], [348], [349], [350], [351], [352], [353], [354], [355], [356], [357], [358], [359], [360], [361], [362], [363], [364], [365], [366], [367], [368], [369], [370], [371], [372], [373], [374], [375], [376], [377], [378], [379], [380], [381], [382], [383], [384], [385], [386], [387], [388], [389], [390], [391], [392], [393], [394], [395], [396], [397], [398], [399], [400], [401], [402], [403], [404], [405], [406], [407], [408], [409], [410], [411], [412], [413], [414], [415], [416], [417], [418], [419], [420], [421], [422], [423], [424], [425], [426], [427], [428], [429], [430], [431], [432], [433], [434], [435], [436], [437], [438], [439], [440], [441], [442], [443], [444], [445], [446], [447], [448], [449], [450], [451], [452], [453], [454], [455], [456], [457], [458], [459], [460], [461], [462], [463], [464], [465], [466], [467], [468], [469], [470], [471], [472], [473], [474], [475], [476], [477], [478], [479], [480], [481], [482], [483], [484], [485], [486], [487], [488], [489], [490], [491], [492], [493], [494], [495], [496], [497], [498], [499], [500], [501], [502], [503], [504], [505], [506], [507], [508], [509], [510], [511], [512], [513], [514], [515], [516], [517], [518], [519], [520], [521], [522], [523], [524], [525], [526], [527], [528], [529], [530], [531], [532], [533], [534], [535], [536], [537], [538], [539], [540], [541], [542], [543], [544], [545], [546], [547], [548], [549], [550], [551], [552], [553], [554], [555], [556], [557], [558], [559], [560], [561], [562], [563], [564], [565], [566], [567], [568], [569], [570], [571], [572], [573], [574], [575], [576], [577], [578], [579], [580], [581], [582], [583], [584], [585], [586], [587], [588], [589], [590], [591], [592], [593], [594], [595], [596], [597], [598], [599], [600], [601], [602], [603], [604], [605], [606], [607], [608], [609], [610], [611], [612], [613], [614], [615], [616], [617], [618], [619], [620], [621], [622], [623], [624], [625], [626], [627], [628], [629], [630], [631], [632], [633], [634], [635], [636], [637], [638], [639], [640], [641], [642], [643], [644], [645], [646], [647], [648], [649], [650], [651], [652], [653], [654], [655], [656], [657], [658], [659], [660], [661], [662], [663], [664], [665], [666], [667], [668], [669], [670], [671], [672], [673], [674], [675], [676], [677], [678], [679], [680], [681], [682], [683], [684], [685], [686], [687], [688], [689], [690], [691], [692], [693], [694], [695], [696], [697], [698], [699], [700], [701], [702], [703], [704], [705], [706], [707], [708], [709], [710], [711], [712], [713], [714], [715], [716], [717], [718], [719], [720], [721], [722], [723], [724], [725], [726], [727], [728], [729], [730], [731], [732], [733], [734], [735], [736], [737], [738], [739], [740], [741], [742], [743], [744], [745], [746], [747], [748], [749], [750], [751], [752], [753], [754], [755], [756], [757], [758], [759], [760], [761], [762], [763], [764], [765], [766], [767], [768], [769], [770], [771], [772], [773], [774], [775], [776], [777], [778], [779], [780], [781], [782], [783], [784], [785], [786], [787], [788], [789], [790], [791], [792], [793], [794], [795], [796], [797], [798], [799], [800], [801], [802], [803], [804], [805], [806], [807], [808], [809], [810], [811], [812], [813], [814], [815], [816], [817], [818], [819], [820], [821], [822], [823], [824], [825], [826], [827], [828], [829], [830], [831], [832], [833], [834], [835], [836], [837], [838], [839], [840], [841], [842], [843], [844], [845], [846], [847], [848], [849], [850], [851], [852], [853], [854], [855], [856], [857], [858], [859], [860], [861], [862], [863], [864], [865], [866], [867], [868], [869], [870], [871], [872], [873], [874], [875], [876], [877], [878], [879], [880], [881], [882], [883], [884], [885], [886], [887], [888], [889], [890], [891], [892], [893], [894], [895], [896], [897], [898], [899], [900], [901], [902], [903], [904], [905], [906], [907], [908], [909], [910], [911], [912], [913], [914], [915], [916], [917], [918], [919], [920], [921], [922], [923], [924], [925], [926], [927], [928], [929], [930], [931], [932], [933], [934], [935], [936], [937], [938], [939], [940], [941], [942], [943], [944], [945], [946], [947], [948], [949], [950], [951], [952], [953], [954], [955], [956], [957], [958], [959], [960], [961], [962], [963], [964], [965], [966], [967], [968], [969], [970], [971], [972], [973], [974], [975], [976], [977], [978], [979], [980], [981], [982], [983], [984], [985], [986], [987], [988], [989], [990], [991], [992], [993], [994], [995], [996], [997], [998], [999]]
    # groups_random = help.groups_random_create(Dim, 25, 10)
    groups_one = help.groups_one_create(Dim)
    # print('random grouping: ', groups_random)
    #
    # simple_problems_Dim, simple_problems_Data_index = help.extract(groups_modified)
    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the grouping strategy firstly do the best features combination in initial population
    simple_population_size = 20
    complex_population_size = 1000
    simple_MAX_iteration = 1000
    complex_MAX_iteration = 50000
    draw_simple_Max_iteration = 500
    draw_complex_Max_iteration = 2000

    test_times = 1
    efficient_Lasso_iteration_times = 0
    # efficient_random_iteration_times = 0
    efficient_one_iteration_times = 0
    efficient_complex_iteration_times = 0

    max_or_min = 1

    # print(init_population)
    index = [0] * Dim
    best_simple = 0

    simple_Lasso_problems_trace = []
    # simple_random_problems_trace = []
    simple_one_problems_trace = []
    complex_problems_trace = []

    best_Lasso_index_trace = []
    # best_random_index_trace = []
    best_one_index_trace = []

    simple_Lasso_problems_trace_average = []
    # simple_random_problems_trace_average = []
    simple_one_problems_trace_average = []
    complex_problems_trace_average = []

    best_Lasso_index_average = []
    # best_random_index_average = []
    best_one_index_average = []

    time_Lasso_group = 0
    # time_random_group = 0
    time_one_group = 0
    time_normal = 0
    for t in range(test_times):
        print('round', t + 1)
        time1 = time.process_time()
        best_Lasso_obj_trace, best_Lasso_index, e_Lasso_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                                  simple_MAX_iteration,
                                                                                  benchmark_function, scale_range,
                                                                                  evaluate_function, groups_Lasso,
                                                                                  max_or_min)
        time2 = time.process_time()
        efficient_Lasso_iteration_times += e_Lasso_time
        simple_Lasso_problems_trace.append(best_Lasso_obj_trace)

        best_Lasso_index_trace.append(best_Lasso_index)

        time_Lasso_group += time2 - time1

        # best_random_obj_trace, best_random_index, e_random_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
        #                                                                           simple_MAX_iteration,
        #                                                                           benchmark_function, scale_range,
        #                                                                           evaluate_function, groups_random,
        #                                                                           max_or_min)
        #
        #
        # efficient_random_iteration_times += e_random_time
        # simple_random_problems_trace.append(best_random_obj_trace)
        # best_random_index_trace.append(best_random_index)
        # time_random_group += time3 - time2
        time3 = time.process_time()
        best_one_obj_trace, best_one_index, e_one_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                                  simple_MAX_iteration,
                                                                                  benchmark_function, scale_range,
                                                                                  evaluate_function, groups_one,
                                                                                  max_or_min)

        time4 = time.process_time()
        efficient_one_iteration_times += e_one_time
        simple_one_problems_trace.append(best_one_obj_trace)
        best_one_index_trace.append(best_one_index)
        time_one_group += time4 - time3

        best_complex_trace, e_complex_time = DE.ComplexProblemsOptimization(Dim, complex_population_size, complex_MAX_iteration,
                                                            evaluate_function, benchmark_function, scale_range,
                                                            max_or_min)
        time5 = time.process_time()
        efficient_complex_iteration_times += e_complex_time
        complex_problems_trace.append(best_complex_trace)
        time_normal += time5 - time4

    print('--------------------------------------------------------------------')
    print('Average Lasso group time: ', time_Lasso_group / test_times)
    # print('Average random group time: ', time_random_group / test_times)
    print('Average one group time: ', time_one_group / test_times)
    print('Average normal time: ', time_normal / test_times)
    print('')
    print('Average efficient Lasso group : ', efficient_Lasso_iteration_times / test_times)
    # print('Average efficient random group time: ', efficient_random_iteration_times / test_times)
    print('Average efficient one group time: ', efficient_one_iteration_times / test_times)
    print('Average efficient normal time: ', efficient_complex_iteration_times / test_times)
    simple_Lasso_problems_trace = np.array(simple_Lasso_problems_trace)
    # simple_random_problems_trace = np.array(simple_random_problems_trace)
    simple_one_problems_trace = np.array(simple_one_problems_trace)

    best_Lasso_index_trace = np.array(best_Lasso_index_trace)
    # best_random_index_trace = np.array(best_random_index_trace)
    best_one_index_trace = np.array(best_one_index_trace)

    complex_problems_trace = np.array(complex_problems_trace)

    for i in range(len(simple_Lasso_problems_trace[0])):
        simple_Lasso_problems_trace_average.append(sum(simple_Lasso_problems_trace[:, i]) / test_times)
    # for i in range(len(simple_random_problems_trace[0])):
    #     simple_random_problems_trace_average.append(sum(simple_random_problems_trace[:, i]) / test_times)
    for i in range(len(simple_one_problems_trace[0])):
        simple_one_problems_trace_average.append(sum(simple_one_problems_trace[:, i]) / test_times)

    for i in range(len(complex_problems_trace[0])):
        complex_problems_trace_average.append(sum(complex_problems_trace[:, i]) / test_times)

    for i in range(len(best_Lasso_index_trace[0])):
        best_Lasso_index_average.append(sum(best_Lasso_index_trace[:, i]) / test_times)
    # for i in range(len(best_random_index_trace[0])):
    #     best_random_index_average.append(sum(best_random_index_trace[:, i]) / test_times)
    for i in range(len(best_one_index_trace[0])):
        best_one_index_average.append(sum(best_one_index_trace[:, i]) / test_times)

    help.write_trace(name + '_LASSO', simple_Lasso_problems_trace, simple_Lasso_problems_trace_average)
    # help.write_trace(name + '_random', simple_random_problems_trace, simple_random_problems_trace_average)
    help.write_trace(name + '_one', simple_one_problems_trace, simple_one_problems_trace_average)
    help.write_trace(name + '_normal', complex_problems_trace, complex_problems_trace_average)

    x1 = np.linspace(complex_population_size, complex_population_size * (draw_simple_Max_iteration + 1),
                     draw_simple_Max_iteration, endpoint=False)
    x2 = np.linspace(complex_population_size, complex_population_size * (draw_complex_Max_iteration + 1),
                     draw_complex_Max_iteration, endpoint=False)
    help.draw_obj(x1, x2, simple_Lasso_problems_trace_average[0:draw_simple_Max_iteration],
                  simple_one_problems_trace_average[0:draw_simple_Max_iteration],
                  complex_problems_trace_average[0:draw_complex_Max_iteration], name)

    x = np.linspace(1, Dim + 1, Dim, endpoint=False)
    help.draw_var(x, best_Lasso_index_average, best_one_index_average, index, name)
    statistic = [10, 50, 100, 200]
    for s in statistic:
        print('Kruskal statistic in ', str(s * complex_population_size), 'th Evaluation times: ',
              kruskal(simple_Lasso_problems_trace[:, s-1], simple_one_problems_trace[:, s-1], complex_problems_trace[:, s-1]))

