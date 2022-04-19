
CAN_NUM = 150


def _get_answer_coverage(answer, all_entities):
    if len(answer) == 0:
        return 0.
    if len(all_entities) == 0:
        return 0.
    else:
        found = 0.
        total = len(answer)
        for ans in answer:
            if ans in all_entities:
                found += 1.
        return found / total




if __name__ == '__main__':
    # get total question number
    spo_score_file = "/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/predict/spo_score_file.txt"
    f1 = open(spo_score_file, 'r', encoding='utf-8')
    li = []
    flag = 0
    for line in f1.readlines():
        line = line.strip("\n")
        li.append(line.split(", "))
    QUE_NUM = int(li[-1:][0][0])
    outter_li = []
    for i in range(1, QUE_NUM+1):
        inner_li = []
        for con in li:
            if int(con[0]) == i:
                inner_li.append([con[0], con[3]])
        outter_li.append(inner_li)

    outter = []
    for i in outter_li:
        entity_li = []
        flag = 0
        for con in i:
            if flag < CAN_NUM:
                l = con[1].split(" and ")
                for entity in l[::2]:
                    entity_li.append(entity)
            flag += 1
        outter.append(list(set(entity_li)))
    # for entity in outter:
    #     print(entity, "\n\n\n")

    # get answer
    question_set_file = "/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/predict/question_set.txt"
    f2 = open(question_set_file, 'r', encoding="utf-8")
    answer_li = []
    for line in f2.readlines():
        line = line.strip("\n")
        answer_li.append(line.split(", ")[2:])
    print(answer_li)


    sum = 0
    for i in range(QUE_NUM):
        # print(answer_li[i])
        sum += _get_answer_coverage(answer_li[i], outter[i])
        print("i: ", i)
        print(_get_answer_coverage(answer_li[i], outter[i]), "\n")
    print(sum)
    print("recall is: " + str(sum / len(answer_li)))



