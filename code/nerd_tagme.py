import requests
#import truecase


#MY_GCUBE_TOKEN = 'f08bb655-6465-4cdb-a5b2-7b7195cea1d7-843339462'
MY_GCUBE_TOKEN = '9dc5f6c0-3040-411b-9687-75ca53249072-843339462'


def get_wikipedialink(pageid):
    info_url = "https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids=" + pageid + "&inprop=url&format=json"
    try:
        response = requests.get(info_url)
        result = response.json()["query"]["pages"]
        if result:
            link = result[pageid]['fullurl']
            return link
    except:
        print("get_wikipedialink problem", pageid)


def get_qid(wikipedia_link):
    url = "https://tools.wmflabs.org/openrefine-wikidata/en/api?query=" + wikipedia_link
    try:
        response = requests.get (url)
        results = response.json ()["result"]
        if results:
            qid = results[0]['id']
            wikidata_label = results[0]['name']
            return qid, wikidata_label
    except:
        print ("get_qid problem", wikipedia_link)


def tagme_entity_linking(text):
    # Main method, text annotation with WAT entity linking system
    tagme_url = 'https://tagme.d4science.org/tagme/tag?lang=en&gcube-token=9dc5f6c0-3040-411b-9687-75ca53249072-843339462&text=' + text
    try:
        response = requests.get(tagme_url)
        tagme_annotations = [TagMeAnnotation(a) for a in response.json()['annotations']]
        return [w.json_dict() for w in tagme_annotations]
    except:
        print("here is a timeout error!")
        return None

class TagMeAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['link_probability']
        # annotated text
        self.spot = d['spot']
        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']

    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }

class EntityLinkTagMeMatch():
    def __init__(self, tagme_threshold=0):
        self.tagme_threshold = tagme_threshold

    def get_entities_tagme(self, ques_truecase):
        tagme_ent = self.get_response_tagme(ques_truecase)
        tagme_ent['wikidata'] = []
        for link in tagme_ent['spot']:
            pageid = link[2]
            wikipedia_link = get_wikipedialink(pageid)
            if wikipedia_link:
                try:
                    wikidata_id, wikidata_label = get_qid(wikipedia_link)
                    if wikidata_id:
                        tagme_ent['wikidata'].append((wikidata_id, wikidata_label, link[1]))
                except:
                    print (wikipedia_link)
                    continue
        return tagme_ent

    def get_response_tagme(self, ques):
        tagme_ent = {}
        tagme_ent['spot'] = []
        try:
            annotations = tagme_entity_linking(ques)
            # print (annotations)
            if annotations:
                for doc in annotations:
                    if doc['rho'] >= self.tagme_threshold:
                        doc['spot'] = ques[doc["start"]:doc["end"]]
                        tagme_ent['spot'].append(
                            (doc['spot'], doc['wiki_title'], str(doc['wiki_id']), doc['rho'], doc['start'], doc['end']))
        except:
            print("TAGME Problem \n", ques)
        return tagme_ent


def get_seed_entities_tagme(TAGME, question):
    wiki_ids = set()
    tagme_ent = TAGME.get_entities_tagme(question)
    # print (tagme_ent['wikidata'])
    if 'wikidata' in tagme_ent and 'spot' in tagme_ent:
        for id1 in tagme_ent['wikidata']:
            index = tagme_ent['wikidata'].index(id1)
            text = tagme_ent['spot'][index][0].lower()
            score = float(tagme_ent['spot'][index][3])
            wiki_ids.add((id1[0], score, text, id1[1]))
    # for item in wiki_ids:
    #     print(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2])  + '\t' + str(item[3])  + '\n')
    return wiki_ids

    # # 选score最大的entity，并用列表返回
    # max_score = 0
    # max_li = []
    # for i in wiki_ids:
    #     if i[1] >= max_score:
    #         max_score = i[1]
    #         max_li = list(i)
    #     # print("i[1]", i[1])
    # return max_li
