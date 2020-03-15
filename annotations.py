import numpy as np
import pandas as pd
#frame, title, subject, omlidt, direkte, kortnyt, location

A = np.array([[658, 908, 'to planlagde angreb med drone', 'terror'],
    [999, 1059, 'kåre quist', 'tv avisen'],
    [2640, 2731, 'tore hamming', 'ph.d.-studerende, gæsteforsker, diis'],
    [3140, 3229, 'tore hamming', 'ph.d.-studerende, gæsteforsker, diis'],
    [4139, 4229, 'mette mayli albæk', 'retsanalytiker, dr nyheder'],
    [5889, 5980, 'tore hamming', 'ph.d.-studerende, gæsteforsker, diis'],
    [6111, 6268, 'tråde til angreb mod lars hedegaard', 'terror'],
    [6271, 6395, 'mette mayli albæk', 'retsanalytiker, dr nyheder'],
    [6398, 8427, 'tråde til angreb mod lars hedegaard', 'terror'],
    [8500, 8927, 'dømt for hjælp til selvmord', 'aktiv dødshjælp'],
    [8981, 9197, 'homopingviner kidnapper unge', 'odense zoo'],
    [9983, 10098, 'annamary christiansen', 'kontorassistent'],
    [10633, 10748, 'annamary christiansen', 'kontorassistent'],
    [11132, 11248, 'torben svarrer', 'politiinspektør'],
    [11856, 11996, 'torben svarrer', 'politiinspektør'],
    [12948, 13098, 'torben svarrer', 'politiinspektør'],
    [13781, 13896, 'søren pape poulsen', 'justitsminister, konservative'],
    [14234, 14346, 'annamary christiansen', 'kontorassistent'],
    [14657, 14958, 'trump fortsætter presset mod iran', 'fn'],
    [14961, 15082, 'donald trump', 'præsident, usa'],
    [15086, 15842, 'trump fortsætter presset mod iran', 'fn'],
    [15846, 15968, 'emmanuel macron', 'præsident, frankrig'],
    [16415, 16686, 'trump kræver opbakning fra fn-lande', 'fns sikkerhedsråd'],
    [16690, 16812, 'steffen gram', 'international korrespondent'],
    [16816, 18642, 'trump kræver opbakning fra fn-lande', 'fns sikkerhedsråd'],
    [18713, 19482, 'dømt for hjælp til selvmord', 'aktiv dødshjælp'],
    [20096, 20209, 'lise pedersen', ' it-leverence ansvarlig, roskilde'],
    [20320, 20429, 'lise pedersen', ' it-leverence ansvarlig, roskilde'],
    [21072, 21161, 'lise pedersen', ' it-leverence ansvarlig, roskilde'],
    [21720, 21829, 'svend lings', 'pensioneret læge'],
    [22066, 22187, 'lise pedersen', ' it-leverence ansvarlig, roskilde'],
    [22869, 22984, 'else laursen', 'pensionist'],
    [23047, 23162, 'torben bjerre', 'sygemeldt'],
    [23219, 23335, 'maria juul randers', 'studerende'],
    [23721, 23835, 'gorm greisen', 'formand etisk råd, børnelæge'],
    [24271, 24385, 'lise pedersen', ' it-leverence ansvarlig, roskilde'],
    [24884, 24996, 'liselott blixt', 'formand, sundhedsudvalget, dansk Folkeparti'],
    [28244, 28493, 'deltag i debatten på dr.dk', 'aktiv dødshjælp'],
    [28497, 28792, 'borgere stemmer nej til sluse', 'kerteminde'],
    [28798, 28917, 'kasper ejsing olesen', 'borgmester kerteminde, socialdemokratiet'],
    [28921, 29608, 'borgere stemmer nej til sluse', 'kerteminde'],
    [29689, 29760, 'iben høj', 'arbejdsgruppen for et ansvarligt sluseprojekt'],
    [30011, 30523, 'homopingviner kidnapper unge', 'odense zoo'],
    [30859, 30948, 'mette heikel', 'dyrepasser, odense zoo'],
    [31108, 31173, 'sandie hedegård munck', 'dyrepasser, Odense zoo'],
    [32409, 32547, 'mikkel stelvig', 'zoolog'],
    [32959, 33074, 'mette heikel', 'dyrepasser, odense zoo'],
    [33458, 33573, 'mette heikel', 'dyrepasser, odense zoo'],
    [34036, 34148, 'louise gade', 'dr vejret']])

direkte = np.array([[6120, 8420, 'københavn'],
                   [16404, 18641, 'new york']])

omlidt = np.array([[8500, 8933], [8981, 9202]])

ret_arr = []
for a in A:
    i = int(a[0])
    end = int(a[1])
    while i <= end:
        ret_arr.append([i, a[2], a[3], False, False, False, None])
        i += 1

ret_arr =np.array(ret_arr)
for o in omlidt:
    i = int(o[0])
    end = int(o[1])
    while i <= end:
        ret_arr[ret_arr[:,0] == i, 3] = True
        i += 1


dires = []
for d in direkte:
    i = int(d[0])
    end = int(d[1])
    while i <= end:
        v = ret_arr[ret_arr[:,0] == i]
        if v.size == 0:
            dires.append([i, None, None, False, True, False, d[2]])
        else:
            ret_arr[ret_arr[:,0] == i, 4] = True
            ret_arr[ret_arr[:,0] == i, 6] = d[2]
        i +=1

merged = np.append(ret_arr, dires).reshape(ret_arr.shape[0]+len(dires),7)
data = pd.DataFrame(merged)
data = data.rename(columns={0: "Frame", 1: "Story title", 2: "Story subject", 3: "Om lidt", 4: "Direkte", 5: "Kort nyt", 6: "Location"})
data = data.sort_values(by=['Frame'])
data.to_csv("annotated_data1", index=False)
#np.save("annos", ret_arr)
