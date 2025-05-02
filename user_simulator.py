import random
import json
from typing import Dict, List, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd

class user_simulator:
    def __init__(self, meta, llm):
        self.meta = meta
        self.llm = llm
        self.retrieval_result = []
        self.retrieval_reciprocal_rank = []

    def initial_ambiguous_query(self):
        """
        generates an ambiguous query, which contains partial information about the wanted item
        """
        # in this prototype, we simply invoke an llm to generate an ambiguous initial query
        ambiguous_query_prompt = PromptTemplate(
            input_variables=["meta"],
            template=(
                "You are a user who is looking for a product on an e-commerce website such as Amazon. "
                "You probably know what you want, but you are not sure about the exact name or description. "
                "Although you are given the full name of the product, you cannot return it as the query. "
                "Your job is to generate a query that is still ambiguous, but contains key partial information about the wanted item. "
                "This resembles a real user query that is not too specific and does not contain the full name of the product. "
                "For example, if the product is a 'Samsung Galaxy S21 silver smartphone with 128GB storage', "
                "you may return 'Galaxy S21' or 'Samsung smartphone' for example. "
                "you may use the product title and some features(if any) to generate the query, which is at most two to five words. "
                "The product title is {meta[title]} and the features are {meta[features]} and {meta[description]}. "
                "Please return the query in a single line without any additional text or explanation. "
                "The query should be a short phrase(2-5 words) and should not contain punctuation or special characters. "
            ),
        )
        # generate the query
        ambiguous_query = ambiguous_query_prompt.format(meta=self.meta)
        return ambiguous_query

    def answer_clarification_question(self, question_str):
        """
        generates an answer to a clarification question, which is asked by the system
        """
        answer_clarification_question_prompt = PromptTemplate(
            input_variables=["meta", "question"],
            template=(
                "You are a user who is looking for a product on an e-commerce website such as Amazon. "
                "You had already made an initial query, which only contains partial information about the wanted item. "
                "The system is asking you a clarification question to help you find the product. "
                "Your job is to answer the question to help the system better understand your needs. "
                "You are not allowed to return the full name of the product, since we're now simulating a real user query scenario. "
                "The question is: {question}. "
                "You may refer to the product title and some features(if any) to answer the question. "
                "The product title is {meta[title]} and the features are {meta[features]} and {meta[description]}. "
                "Please return the answer in a single line without any additional text or explanation. "
            ),
        )
        answer = answer_clarification_question_prompt.format(
            meta=self.meta, question=question_str
        )
        return answer

    def eval_retrieval(self, retrieved_items, k=10):
        """
        evaluates the retrieval result
        """
        # retrieval_result is a list of tuples, where each tuple contains the id, the description, and the retrieval score
        # we need to first sort the list by the retrieval score
        retrieved_items = sorted(
            retrieved_items, key=lambda x: x[2], reverse=True
        )

        # first check the rank for MRR
        found = False
        for i, item in enumerate(retrieved_items):
            if item[0] == self.meta["parent_asin"]:
                self.retrieval_reciprocal_rank.append(1/(i + 1))
                found = True
                break
        if not found:
            self.retrieval_reciprocal_rank.append(0)

        # we only check the top k items(to compute Hit@k)
        # in this prototype, we use k=10
        retrieved_items = retrieved_items[:k]

        for item in retrieved_items:
            if item[0] == self.meta["parent_asin"]:
                self.retrieval_result.append(True)
                return
        self.retrieval_result.append(False)
        return

    def get_result(self):
        """
        returns the retrieval result
        """
        return self.retrieval_result, self.retrieval_reciprocal_rank


def accumulate_retrieval_result(retrieval_result_list, retrieval_reciprocal_rank_list):
    """
    accumulates the retrieval result
    """
    # retrieval_result_list is a list of lists, where each list contains a series of retrieval result for a single item
    # for example, it could be [False, False, True, True, True]
    # Each retrieval result may have a different length

    # first, count the lengths of each retrieval result
    # this is used to evaluate the performance of 'early stopping'
    retrieval_result_length = {}
    for retrieval_result in retrieval_result_list:
        # if a given length is not in the dictionary, we add it by setting it to 1
        # otherwise, we increment the count of that length by 1
        retrieval_result_length[len(retrieval_result)] = retrieval_result_length.get(
            len(retrieval_result), 0
        ) + 1

    # now, we want to compute hit@k (k = 4 by default in our implementation) for each retrieval turn
    # for example, we check the first turn from all retrieval results, and aggregate them

    # we need to find the maximum length of the retrieval result
    max_length = max(retrieval_result_length.keys())

    # we pad the retrieval result list with None
    # so that all retrieval results have the same length
    for retrieval_result in retrieval_result_list:
        retrieval_result.extend([None] * (max_length - len(retrieval_result)))

    for reciprocal_rank in retrieval_reciprocal_rank_list:
        reciprocal_rank.extend([None] * (max_length - len(reciprocal_rank)))

    # now, we need to aggregate the retrieval results per turn
    # each turn may contain a different number of retrieval results(so we need to ignore None and only count True/False)
    hitmiss = [[0, 0] for _ in range(max_length)]
    # hitmiss[i][0] is the number of True for the i-th retrieval turn
    # hitmiss[i][1] is the number of False for the i-th retrieval turn

    for retrieval_result in retrieval_result_list:
        for i in range(max_length):
            if retrieval_result[i] is None:
                continue

            if retrieval_result[i]:
                hitmiss[i][0] += 1
            else:
                hitmiss[i][1] += 1

    hit_at_k_for_each_turn = [0.0 for _ in range(max_length)]

    for i in range(max_length):
        # it is guaranteed that hitmiss[i][0] + hitmiss[i][1] > 0
        hit_at_k_for_each_turn[i] = hitmiss[i][0] / (hitmiss[i][0] + hitmiss[i][1])


    # We also compute MRR for each turn
    MRR_per_turn = [0.0 for _ in range(max_length)]
    MRR_count = [0 for _ in range(max_length)]

    for reciprocal_rank in retrieval_reciprocal_rank_list:
        for i in range(max_length):
            if reciprocal_rank[i] is None:
                continue
            else:
                MRR_per_turn[i] += reciprocal_rank[i]
                MRR_count[i] += 1

    for i in range(max_length):
        # it is guaranteed that MRR_count[i] > 0
        MRR_per_turn[i] /= MRR_count[i]

    return retrieval_result_length, hit_at_k_for_each_turn, MRR_per_turn
