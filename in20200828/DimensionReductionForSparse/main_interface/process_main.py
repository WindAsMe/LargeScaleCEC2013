import numpy as np
from in20200828.DimensionReductionForSparse.util import help

if __name__ == '__main__':
    m = [
        [2.7962302213315837e+17, 2.7962302213315837e+17, 2.7962301845187766e+17, 2.7962301845187766e+17,
         2.7962301845187766e+17, 2.7962301845187766e+17, 2.7962301845187766e+17, 2.7962301845187766e+17,
         2.746829270103228e+17, 2.746829270103228e+17, 2.746829270103228e+17, 2.746829270103228e+17,
         2.746829270103228e+17, 2.746829270103228e+17, 2.746829270103228e+17, 2.7030529682957568e+17,
         2.7030529682957568e+17, 2.7030529682957568e+17, 2.7030529682957568e+17, 2.7030529682957568e+17,
         2.7030529679203984e+17, 2.7030529679203984e+17, 2.7030529679203984e+17, 2.7030529679203984e+17,
         2.7030529679203984e+17, 2.703052938592313e+17, 2.703052938592313e+17, 2.703052938592313e+17,
         2.703052938592313e+17, 2.703052938592313e+17]
        ,[
            2.8437794273924755e+17, 2.8437794273924755e+17, 2.8437794273924755e+17, 2.8437794273924755e+17, 2.791697862278001e+17, 2.7916977490161536e+17, 2.7916977490161536e+17, 2.7916977490161536e+17, 2.7916977490161536e+17, 2.7916977490161536e+17, 2.7916977490161536e+17, 2.7916977490161536e+17, 2.7916976653285482e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17, 2.7916976357574134e+17]
        ,[
            2.94396550023987e+17, 2.7426256045253978e+17, 2.7426253953844214e+17, 2.7426253953844214e+17, 2.7426253953844214e+17, 2.7426253953844214e+17, 2.7426253953844214e+17, 2.7426253953844214e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17, 2.7331545717451776e+17]
        ,[
            2.8722652061644723e+17, 2.8722651897667146e+17, 2.8722651897667146e+17, 2.87226516881143e+17, 2.8722651585640054e+17, 2.8722651585640054e+17, 2.8722651585640054e+17, 2.8722651585640054e+17, 2.8722651585640054e+17, 2.87154210209635e+17, 2.87154210209635e+17, 2.87154210209635e+17, 2.87154210209635e+17, 2.87154210209635e+17, 2.87154210209635e+17, 2.87154210209635e+17, 2.836103236707288e+17, 2.804560268909537e+17, 2.804560268765278e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17, 2.8045602659084272e+17]
        ,[
            2.9524450460220486e+17, 2.9524450460220486e+17, 2.9524450460220486e+17, 2.9524450460220486e+17, 2.9524447567421606e+17, 2.9524447567421606e+17, 2.889963291414359e+17, 2.8899632912391174e+17, 2.8899632636695776e+17, 2.8899632091922426e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.8899631937950074e+17, 2.887022776173233e+17, 2.887022776173233e+17, 2.887022776173233e+17, 2.887022776173233e+17, 2.887022776173233e+17, 2.887022776173233e+17]
        ,[
            2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5685343504841344e+17, 2.5629920687794144e+17, 2.5629920687794144e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17, 2.5629919425230685e+17]
        ,[
            2.6510931093444422e+17, 2.6510931093444422e+17, 2.6510931093444422e+17, 2.6510931093444422e+17, 2.6501408463025376e+17, 2.650140842049448e+17, 2.650140842049448e+17, 2.650140842049448e+17, 2.650140842049448e+17, 2.650140842049448e+17, 2.650140842049448e+17, 2.650140834565122e+17, 2.6477732898941338e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17, 2.647773274205066e+17]
        ,[
            2.936336373883449e+17, 2.936336362039566e+17, 2.936336362039566e+17, 2.936336362039566e+17, 2.936336362039566e+17, 2.9363363240727763e+17, 2.9363363240727763e+17, 2.9363363240727763e+17, 2.9363363240727763e+17, 2.9363363240727763e+17, 2.9363363240727763e+17, 2.9363363240727763e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17, 2.89483123708076e+17]
        ,[
            2.9725419117267936e+17, 2.972541909230846e+17, 2.972541846107723e+17, 2.97215189243241e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17, 2.9721518786924576e+17]
        ,[
            2.5490623432090794e+17, 2.549062343080176e+17, 2.549062332972527e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5490623324846368e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17, 2.5484806225500883e+17]
    ]
    m = np.array(m)
    ave = []
    for i in range(len(m[0])):
        ave.append(np.mean(m[:, i]))
    x = np.linspace(0, 3000000, 30)
    # print(x)
    help.draw_obj(x, ave, 'temp')
    print(np.mean(m[:, 29]))
    print(np.std(m[:, 29], ddof=1))
