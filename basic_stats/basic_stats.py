import json
import numpy as np
import matplotlib.pyplot as plt

def get_robust_f1(true_positive, total_prediction, total_gold):
    epsilon = 1e-6
    p = true_positive / (total_prediction+epsilon)
    r = true_positive / (total_gold+epsilon)
    f = 2*p*r/(p+r+epsilon)
    return f, p, r

def spanstr2intpair(spanstr):
    try:
        tmp = spanstr[1:-1].split(',')
        start, end = int(tmp[0]), int(tmp[1])
    except:
        start, end = -1, -1
    return start, end

def has_indices(list_of_indices, query_indices, relaxed=False):
    """
    E.g.,
    "list_of_indices": [
                "(14,23)",
                "(37,43)",
                "(58,63)",
                "(67,73)",
                "(151,155)",
                "(208,212)",
                "(225,230)",
                "(271,280)",
                "(302,307)"
              ]
    """
    if not relaxed:
        if query_indices in list_of_indices:
            return True, query_indices
        return False, ''
    # relaxed means returning true as long as finding overlap
    query_indices_start, query_indices_end = spanstr2intpair(query_indices)
    for indices in list_of_indices:
        indices_start, indices_end = spanstr2intpair(indices)
        if indices_start < query_indices_end and indices_end > query_indices_start:
            return True, indices
    return False, ''

def questionTextReplace(question):
    if question=='What event has already happened?' or question=='What event has already finished?':
        return 'What events have already finished?'
    if question=='What is happening now?' or question=='What event has begun but has not finished?':
        return 'What events have begun but has not finished?'
    return question
    # Counter({'What will happen in the future?': 586, 'What event has begun but has not finished?': 550, 'What event has already finished?': 549, 'What event has already happened?': 42, 'What is happening now?': 42})

class AnnotatedPassage:
    num_all_events = []
    num_all_tokens = 0
    num_all_questions = 0
    num_all_default_questions = 0
    num_all_derived_questions = 0
    num_all_answered_questions = 0
    num_all_answers = []
    num_all_answers_default_q = []
    num_all_answers_userprovided = []
    num_all_answers_derived_q = []
    num_all_answers_userprovided_original = []
    num_all_answers_userprovided_derived = []

    def __init__(self, raw_annotated_passage, isTestset=False, verbose=True):
        self.raw_annotated_passage = raw_annotated_passage
        self.passage = raw_annotated_passage['passage']
        self.events = raw_annotated_passage['events'][0] if not isTestset else raw_annotated_passage['events']
        self.passage_id = self.events['passageID']
        tmp = self.passage_id.split("_sentid_")
        self.doc_id = tmp[0]
        self.sent_id = int(tmp[1])
        self.verbose = verbose

        if not isTestset:
            self.question_answer_pairs = raw_annotated_passage['question_answer_pairs']
            for qa in self.question_answer_pairs:
                qa['question'] = questionTextReplace(qa['question'])
                qa['derived_from_question'] = ''
        else:
            self.question_answer_pairs = []
            for questionText, qa in raw_annotated_passage['question_answer_pairs'].items():
                questionText = questionTextReplace(questionText)
                tmp = {'question':questionText}
                tmp.update(qa)
                self.question_answer_pairs.append(tmp)

        self.num_events = len(self.events['answer']['spans'])
        self.num_tokens = len(self.passage.split(' '))
        self.num_questions = len(self.question_answer_pairs)
        self.num_default_questions = 0
        self.num_derived_questions = 0
        self.num_answered_questions = 0
        self.num_answers = []
        self.num_answers_default_q = []
        self.num_answers_derived_q = []

        for qa in self.question_answer_pairs:
            num_answers = len(qa['answer']['spans'])
            isByUser = 'is_default_question' not in qa or not qa['is_default_question']
            if isByUser:
                isDerived = 'derived_from' in qa and qa['derived_from'] is not None
                if isDerived and not isTestset:
                    tmp = self.get_question(qa['derived_from'])
                    if tmp:
                        qa['derived_from_question'] = tmp['question']
            else:
                isDerived = qa['question'].lower() not in \
                            {'what event has already happened?',
                             'what event has already finished?',
                             'what events have already finished?'}
                if isDerived and not isTestset:
                    tmp = self.get_ith_default_q(0)
                    if tmp:
                        qa['derived_from_question'] = tmp['question']

            # Counter({'What will happen in the future?': 586, 'What event has begun but has not finished?': 550, 'What event has already finished?': 549, 'What event has already happened?': 42, 'What is happening now?': 42})

            if 'is_default_question' in qa and qa['is_default_question']:
                self.num_default_questions += 1
                self.num_answers_default_q.append(num_answers)
            if 'derived_from' in qa and qa['derived_from'] is not None:
                self.num_derived_questions += 1
                self.num_answers_derived_q.append(num_answers)
            if 'isAnswered' in qa and qa['isAnswered']:
                self.num_answered_questions += 1
            self.num_answers.append({'nums':num_answers, 'isByUser':isByUser, 'isDerived':isDerived})

        AnnotatedPassage.num_all_events.append(self.num_events)
        AnnotatedPassage.num_all_tokens += self.num_tokens
        AnnotatedPassage.num_all_questions += self.num_questions
        AnnotatedPassage.num_all_default_questions += self.num_default_questions
        AnnotatedPassage.num_all_derived_questions += self.num_derived_questions
        AnnotatedPassage.num_all_answered_questions += self.num_answered_questions
        AnnotatedPassage.num_all_answers += self.num_answers
        AnnotatedPassage.num_all_answers_default_q += self.num_answers_default_q
        AnnotatedPassage.num_all_answers_derived_q += self.num_answers_derived_q

    # def has_event_indices(self, query_indices, relaxed=False):
    #     if not relaxed:
    #         return query_indices in self.events['answer']['indices']
    #     # relaxed means returning true as long as finding overlap
    #     query_indices_start, query_indices_end = spanstr2intpair(query_indices)
    #     for indices in self.events['answer']['indices']:
    #         indices_start, indices_end = spanstr2intpair(indices)
    #         if indices_start < query_indices_end and indices_end > query_indices_start:
    #             return True
    #     return False

    def compareEventToPassage(self, another_passage, relaxed=False):
        num_same = 0
        for event_ix in self.events['answer']['indices']:
            found, _ = has_indices(another_passage.events['answer']['indices'], event_ix, relaxed=relaxed)
            if found:
                num_same += 1
        return num_same

    def eventF1TwoPassages(self, another_passage, relaxed=False):
        tp = self.compareEventToPassage(another_passage, relaxed=relaxed)
        f, p, r = get_robust_f1(tp, self.num_events, another_passage.num_events)
        # p = tp/self.num_events
        # r = tp/another_passage.num_events
        # f = 2*p*r/(p+r)
        return f, tp

    @staticmethod
    def answerF1TwoQuestions(question1, question2, relaxed=False):
        tp = 0
        for answer_ix in question1['answer']['indices']:
            found, _ = has_indices(question2['answer']['indices'], answer_ix, relaxed=relaxed)
            if found:
                tp += 1
        f, p, r = get_robust_f1(tp, len(question1['answer']['indices']), len(question2['answer']['indices']))
        # p = tp/len(question1['answer']['indices'])
        # r = tp/len(question2['answer']['indices'])
        # f = 2*p*r/(p+r)
        return f, tp

    def get_ith_default_q(self, i):
        for qa in self.question_answer_pairs:
            question_id = qa['question_id']
            if question_id.endswith('question-d'+str(i)):
                return qa
        if self.verbose:
            print(f"""There's no {i}-th default question.""")
        return None

    def get_question(self, question_id):
        for qa in self.question_answer_pairs:
            qid = qa['question_id']
            if qid==question_id: return qa
        if self.verbose:
            print(f"""There's no question {question_id}.""")
        return None

# class AnnotatedDocument:
#     def __init__(self):
#         self.annotated_passages = []

def get_unique_passages(list_of_annotated_passages):
    unique_passage_ids_map = {}
    for ann in list_of_annotated_passages:
        if ann.passage_id not in unique_passage_ids_map:
            unique_passage_ids_map[ann.passage_id] = []
        unique_passage_ids_map[ann.passage_id].append(ann)
    return unique_passage_ids_map

def load_annotations(fname, workerIds=None, isTestset=False, verbose=True):
    if isTestset:
        return load_annotations_testset(fname)
    with open(fname) as f:
        annotations = json.load(f)
    annotated_passages = []
    for ann in annotations:
        assignment_id = ann['assignment_id']
        hit_id = ann['hit_id']
        worker_id = ann['worker_id']
        if workerIds and worker_id not in workerIds: continue
        submission_time = ann['submission_time']
        feedback = ann['feedback']
        for raw_annotated_passage in ann['passages']:
            annotated_passage = AnnotatedPassage(raw_annotated_passage,verbose=verbose)
            annotated_passage.assignment_id = assignment_id
            annotated_passage.hit_id = hit_id
            annotated_passage.worker_id = worker_id
            annotated_passage.submission_time = submission_time
            annotated_passage.feedback = feedback

            annotated_passages.append(annotated_passage)
    return annotated_passages

def load_annotations_testset(fname,verbose=True):
    with open(fname) as f:
        annotations = json.load(f)
    annotated_passages = []
    for ann in annotations.values():
        assignment_id = ''
        hit_id = ''
        worker_id = ''
        submission_time = ''
        feedback = ''

        annotated_passage = AnnotatedPassage(ann, isTestset=True,verbose=verbose)
        annotated_passage.assignment_id = assignment_id
        annotated_passage.hit_id = hit_id
        annotated_passage.worker_id = worker_id
        annotated_passage.submission_time = submission_time
        annotated_passage.feedback = feedback

        annotated_passages.append(annotated_passage)
    return annotated_passages


if __name__=='__main__':

    real_fnames = ['../data/train.json']
    annotated_passages = []
    for fnames in real_fnames:
        annotated_passages += load_annotations(fnames)
    docid_map = {}
    worker_map = {}
    for annotated_passage in annotated_passages:
        doc_id = annotated_passage.doc_id
        worker_id = annotated_passage.worker_id
        if doc_id not in docid_map:
            docid_map[doc_id] = []
        if worker_id not in worker_map:
            worker_map[worker_id] = []
        docid_map[doc_id].append(annotated_passage)
        worker_map[worker_id].append(annotated_passage)

    unique_passage_ids_map = get_unique_passages(annotated_passages)
    # basic stats
    print(f"""Total number of:\n"""
          f"""\tPassages = {len(annotated_passages)}\n"""
          f"""\tPassages (deduplicated) = {len(unique_passage_ids_map)}\n"""
          f"""\tEvents = {sum(AnnotatedPassage.num_all_events)}\n"""
          f"""\tEvents/Passage = {sum(AnnotatedPassage.num_all_events)/len(annotated_passages):.2f}\n"""
          f"""\tTokens = {AnnotatedPassage.num_all_tokens}\n"""
          f"""\tTokens/Passage = {AnnotatedPassage.num_all_tokens/len(annotated_passages):.2f}\n""")

    print(f"""---------- Overall ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp)/len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp)/len(tmp):.2f}\n""")

    print(f"""---------- Warm-up ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if not x['isByUser']]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp)/len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp)/len(tmp):.2f}\n""")

    print(f"""---------- Warm-up: Original ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if not x['isByUser'] and not x['isDerived']]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp) / len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp) / len(tmp):.2f}\n""")

    print(f"""---------- Warm-up: Derived ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if not x['isByUser'] and x['isDerived']]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp) / len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp) / len(tmp):.2f}\n""")

    print(f"""---------- Non-warmup ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if x['isByUser']]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp)/len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp)/len(tmp):.2f}\n""")

    print(f"""---------- Non-warmup: Original ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if x['isByUser'] and not x['isDerived']]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp) / len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp) / len(tmp):.2f}\n""")

    print(f"""---------- Non-warmup: Derived ----------""")
    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if x['isByUser'] and x['isDerived']]
    print(f"""\tQuestions = {len(tmp)}\n"""
          f"""\tQuestions/Passage = {len(tmp) / len(annotated_passages):.2f}\n"""
          f"""\tAnswers = {sum(tmp)}\n"""
          f"""\tAnswers/Question = {sum(tmp) / len(tmp):.2f}\n""")

    # use duplicated passages to calculate event agreement and default question agreement
    duplicated_passage_ids = set()
    for pid, passages in unique_passage_ids_map.items():
        if len(passages) > 1:
            duplicated_passage_ids.add(pid)

    print(f"""Passages annotated by multiple workers: {len(duplicated_passage_ids)}""")
    all_pair_event_f1 = []
    all_pair_event_f1_relaxed = []
    # all_pair_def_q1_f1 = []
    all_pair_def_q1_f1_relaxed = []
    # all_pair_def_q2_f1 = []
    all_pair_def_q2_f1_relaxed = []
    # all_pair_def_q3_f1 = []
    all_pair_def_q3_f1_relaxed = []
    default_answer_th = 3
    for pid in duplicated_passage_ids:
        passages = unique_passage_ids_map[pid]
        pair_f1 = []
        pair_f1_relaxed = []
        for ix1, passage1 in enumerate(passages):
            for ix2, passage2 in enumerate(passages):
                if ix1>=ix2:
                    continue
                event_f1, _ = passage1.eventF1TwoPassages(passage2, relaxed=False)
                event_f1_relaxed, _ = passage1.eventF1TwoPassages(passage2, relaxed=True)
                all_pair_event_f1.append(event_f1)
                all_pair_event_f1_relaxed.append(event_f1_relaxed)

                if len(passage1.get_ith_default_q(0)['answer']['indices']) + len(passage2.get_ith_default_q(0)['answer']['indices'])>default_answer_th:
                    def_q1_f1_relaxed, _ = AnnotatedPassage.answerF1TwoQuestions(passage1.get_ith_default_q(0),
                                                                                 passage2.get_ith_default_q(0),
                                                                                 relaxed=True)
                    all_pair_def_q1_f1_relaxed.append(def_q1_f1_relaxed)

                if len(passage1.get_ith_default_q(1)['answer']['indices']) + len(passage2.get_ith_default_q(1)['answer']['indices'])>default_answer_th:
                    def_q2_f1_relaxed, _ = AnnotatedPassage.answerF1TwoQuestions(passage1.get_ith_default_q(1),
                                                                                 passage2.get_ith_default_q(1),
                                                                                 relaxed=True)
                    all_pair_def_q2_f1_relaxed.append(def_q2_f1_relaxed)

                if len(passage1.get_ith_default_q(2)['answer']['indices']) + len(passage2.get_ith_default_q(2)['answer']['indices'])>default_answer_th:
                    def_q3_f1_relaxed, _ = AnnotatedPassage.answerF1TwoQuestions(passage1.get_ith_default_q(2),
                                                                                 passage2.get_ith_default_q(2),
                                                                                 relaxed=True)
                    all_pair_def_q3_f1_relaxed.append(def_q3_f1_relaxed)
                # pair_f1.append(f1)
                # pair_f1_relaxed.append(f1_relaxed)
        # print(pair_f1)
    print(f"""Macro-average of pairwise event f1-scores: {100*np.mean(all_pair_event_f1): .2f}%""")
    print(f"""Macro-average of pairwise event f1-scores (relaxed): {100*np.mean(all_pair_event_f1_relaxed): .2f}%""")
    print(f"""Macro-average of pairwise default Q1 f1-scores (relaxed): {100*np.mean(all_pair_def_q1_f1_relaxed): .2f}% [default_answer_th={default_answer_th}]""")
    print(f"""Macro-average of pairwise default Q2 f1-scores (relaxed): {100*np.mean(all_pair_def_q2_f1_relaxed): .2f}% [default_answer_th={default_answer_th}]""")
    print(f"""Macro-average of pairwise default Q3 f1-scores (relaxed): {100*np.mean(all_pair_def_q3_f1_relaxed): .2f}% [default_answer_th={default_answer_th}]""")


    # plots

    worker_psg_cnt = []
    doc_psg_cnt = []
    for doc_id in docid_map:
        doc_psg_cnt.append(len(docid_map[doc_id]))
    for worker_id in worker_map:
        worker_psg_cnt.append(len(worker_map[worker_id]))
    worker_psg_cnt.sort()
    doc_psg_cnt.sort()

    gini = 1-sum(np.cumsum(worker_psg_cnt))/len(worker_psg_cnt)/sum(worker_psg_cnt)*2
    plt.rcParams.update({'font.size': 13})
    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()
    ax_f.plot(worker_psg_cnt,'k')
    ax_f.set_xlim(0,60)
    ax_f.set_ylim(0,170)
    ax_c.set_ylim(0,170.0/sum(worker_psg_cnt)*100)
    ax_c.figure.canvas.draw()
    ax_f.set_ylabel('#Passages')
    ax_c.set_ylabel('%')
    ax_f.set_xlabel(f'''{len(worker_psg_cnt)} Individual Annotators''')
    plt.title('#Passages by Each Annotator')
    plt.grid(True)
    plt.savefig('./worker_passage_cnt.pdf')
    plt.clf()

    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()
    ax_f.plot(np.linspace(0,100,len(worker_psg_cnt)),np.cumsum(worker_psg_cnt),'k')
    ax_f.plot([0,100],[0,sum(worker_psg_cnt)],'k-.')
    ax_f.set_xlim(0,100)
    ax_f.set_ylim(0,3200)
    ax_c.set_ylim(0,3200.0/sum(worker_psg_cnt)*100)
    ax_c.figure.canvas.draw()
    ax_f.set_ylabel('#Passages')
    ax_c.set_ylabel('%')
    ax_f.set_xlabel('Percent of Annotators')
    plt.title(f'Accumulated Passages by Each Annotator')
    plt.grid(True)
    plt.savefig('./worker_passage_cum_cnt.pdf')
    plt.clf()
    print()
    print(f"""Gini index of passage distribution over workers={gini: .3f}""")

    # plt.plot(worker_psg_cnt)
    # plt.title('Passages by Each Worker')
    # plt.xlabel('Worker')
    # plt.ylabel('#Passages')
    # # plt.show()
    # plt.grid(True)
    # plt.savefig('./worker_passage_cnt.png')
    # plt.clf()
    #
    # plt.hist(worker_psg_cnt, density=True)
    # plt.grid(True)
    # plt.savefig('./worker_passage_cnt_distribution.png')
    #
    plt.plot(doc_psg_cnt,'k')
    plt.title('Passages Annotated in Each Doc')
    plt.xlabel('Doc')
    plt.ylabel('#Passages')
    # plt.show()
    plt.grid(True)
    plt.savefig('./document_passage_cnt.pdf')
    plt.clf()

    plt.hist(AnnotatedPassage.num_all_events, bins=range(0,25), density=True)
    plt.ylim(0,0.15)
    plt.xlim(0,20)
    plt.xticks(list(range(0,21,2)),list(range(0,21,2)))
    plt.yticks([x/100 for x in range(0,20,5)],list(range(0,20,5)))
    plt.ylabel('%')
    plt.savefig('./event_num_histogram.pdf')
    plt.clf()

    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if not x['isByUser'] and not x['isDerived']]
    plt.hist(tmp, bins=range(18), density=True)
    plt.ylim(0,0.5)
    plt.xlim(0,12)
    plt.yticks([x/100 for x in range(0,60,10)],list(range(0,60,10)))
    plt.ylabel('%')
    plt.savefig('./answer_num_histogram_default_original.pdf')
    plt.clf()

    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if not x['isByUser'] and x['isDerived']]
    plt.hist(tmp, bins=range(18), density=True)
    plt.ylim(0,0.5)
    plt.xlim(0,12)
    plt.yticks([x/100 for x in range(0,60,10)],list(range(0,60,10)))
    plt.ylabel('%')
    plt.savefig('./answer_num_histogram_default_derived.pdf')
    plt.clf()

    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if x['isByUser'] and not x['isDerived']]
    plt.hist(tmp, bins=range(18), density=True)
    plt.ylim(0,0.5)
    plt.xlim(0,12)
    plt.yticks([x/100 for x in range(0,60,10)],list(range(0,60,10)))
    plt.ylabel('%')
    plt.savefig('./answer_num_histogram_user_original.pdf')
    plt.clf()

    tmp = [x['nums'] for x in AnnotatedPassage.num_all_answers if x['isByUser'] and x['isDerived']]
    plt.hist(tmp, bins=range(18), density=True)
    plt.ylim(0,0.5)
    plt.xlim(0,12)
    plt.yticks([x/100 for x in range(0,60,10)],list(range(0,60,10)))
    plt.ylabel('%')
    plt.savefig('./answer_num_histogram_user_derived.pdf')
    plt.clf()

    # common events
    from collections import Counter
    import math
    event_counter = Counter()
    for ann in annotated_passages:
        for event in ann.events['answer']['spans']:
            event_counter[event]+=1
    plt.figure(figsize=(10,6))
    words = 50
    plt.plot([math.log(x[1]) for x in event_counter.most_common(words)],'k')
    plt.xticks(list(range(words)), [x[0] for x in event_counter.most_common(words)], rotation=90)
    plt.yticks([math.log(x) for x in [100,500,2500]],[100,500,2500])
    plt.tight_layout()
    plt.xlim(0,words)
    plt.savefig('./most_common_events.pdf')
    plt.clf()