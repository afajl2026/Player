import torch
from transformers import BertTokenizer, BertForSequenceClassification
from Player import Player

class TeacherPlayer(Player):
    PLAYER_NAME = "Teacher Player"  # name for the player

    def __init__(self):
        super().__init__(self.PLAYER_NAME)
        self.model_path = './finetuned_bert_for_similarity'
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=1)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load the model, load the pretrained model, initialize tokenizer.

    def get_Noun(self, target, hand):
        best_noun = None
        best_score = float('-inf')

        for noun in hand:
            input_text = f"[CLS] {target} [SEP] {noun} [SEP]"
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits.item()  # Get the similarity score for the pair

            if score > best_score:
                best_score = score
                best_noun = noun

        return best_noun

    def choose_card(self, target, hand):
        # Select the index of the card from the cards list that is closest to target
        firstpick = self.get_Noun(target, hand)
        if firstpick is None:
            print(f"Error: first pick index is None")
            return 0  # Fallback to a default card index
        bestCardNdx = hand.index(firstpick)
        return bestCardNdx

    def judge_card(self, target, player_cards):
        # Use the BERT model to judge the best card
        print(f"Judging: target={target}, player_cards={player_cards}")
        judges_pick = self.get_Noun(target, player_cards)
        print(f"Judge picked: {judges_pick}")
        if judges_pick not in player_cards:
            print(f"Error: Judge's pick {judges_pick} is not in player cards {player_cards}")
            return player_cards[0]
        return judges_pick

    def process_results(self, result):
        # Handle results returned from server
        print("Result", result)

if __name__ == '__main__':
    player = TeacherPlayer()
    player.run()

# about 46 hours