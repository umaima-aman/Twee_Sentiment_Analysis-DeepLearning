import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


# Define GUI
class SentimentAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis Inference")
        self.root.geometry("1000x700")

        # Load model and tokenizer
        self.model_path = r"C:\Users\umaima\Contacts\Desktop\23i_8009_DL_HT9\9a\sentiment_model"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Main Layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=20, fill="both", expand=True)

        # Left Frame: Dataset Loading
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side="left", padx=20, fill="both", expand=True)

        # Left Pane Heading
        self.dataset_label = tk.Label(self.left_frame, text="Training Dataset", font=("Arial", 14))
        self.dataset_label.pack(pady=10)

        # Load Dataset Button
        self.load_button = tk.Button(self.left_frame, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(pady=10)

        # Treeview for displaying dataset (tweets and labels)
        self.dataset_tree = ttk.Treeview(self.left_frame, columns=("Text", "True Sentiment"), show="headings", height=15)
        self.dataset_tree.heading("Text", text="Tweet")
        self.dataset_tree.heading("True Sentiment", text="True Sentiment")
        self.dataset_tree.pack(side="left", fill="both", expand=True)

        # Scrollbar for dataset treeview
        self.dataset_scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscroll=self.dataset_scrollbar.set)
        self.dataset_scrollbar.pack(side="right", fill="y")

        # True Sentiment Section (display below clicked tweet)
        self.true_sentiment_label = tk.Label(self.left_frame, text="True Sentiment:", font=("Arial", 12))
        self.true_sentiment_label.pack(pady=5)
        self.true_sentiment = tk.Label(self.left_frame, text="None", font=("Arial", 12))
        self.true_sentiment.pack(pady=10)

        # Right Frame: Test Input Section
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side="left", padx=20, fill="both", expand=True)

        # Test Input Section Heading
        self.test_label = tk.Label(self.right_frame, text="Test Input", font=("Arial", 14))
        self.test_label.pack(pady=10)

        # User Input Text Box for Test
        self.text_input_label = tk.Label(self.right_frame, text="Enter Tweet for Sentiment Analysis:", font=("Arial", 12))
        self.text_input_label.pack(pady=5)
        self.text_input = tk.Entry(self.right_frame, width=50)
        self.text_input.pack(pady=10)

        # Button to Analyze Sentiment
        self.analyze_button = tk.Button(self.right_frame, text="Analyze Sentiment", command=self.analyze_sentiment)
        self.analyze_button.pack(pady=10)

        # Predicted Sentiment Section
        self.predicted_sentiment_label = tk.Label(self.right_frame, text="Predicted Sentiment:", font=("Arial", 12))
        self.predicted_sentiment_label.pack(pady=5)
        self.predicted_sentiment = tk.Label(self.right_frame, text="None", font=("Arial", 12))
        self.predicted_sentiment.pack(pady=10)

        # Sentiment Key Section
        self.key_label = tk.Label(self.right_frame, text="Sentiment Key:", font=("Arial", 12))
        self.key_label.pack(pady=5)

        self.key_description = tk.Label(self.right_frame, text="0 = Bearish, 1 = Bullish, 2 = Neutral", font=("Arial", 12))
        self.key_description.pack(pady=5)

    def load_dataset(self):
        try:
            # Open file dialog to select dataset (CSV file)
            dataset_file = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV Files", "*.csv")])
            if not dataset_file:
                return  # If no file selected, return

            # Load dataset
            self.dataset = pd.read_csv(dataset_file)

            # Update the dataset Treeview with the loaded dataset
            self.update_tree(self.dataset_tree, self.dataset[["text", "label"]])

            messagebox.showinfo("Success", "Dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def update_tree(self, tree, data):
        # Clear existing rows in the treeview
        for row in tree.get_children():
            tree.delete(row)

        # Insert new rows
        for _, row in data.iterrows():
            tree.insert("", "end", values=row.tolist())

        # Bind the treeview selection event to update true sentiment
        self.dataset_tree.bind("<<TreeviewSelect>>", self.on_tweet_select)

    def on_tweet_select(self, event):
        # Get selected tweet's index
        selected_item = self.dataset_tree.selection()
        if selected_item:
            tweet_index = self.dataset_tree.index(selected_item[0])
            true_sentiment = self.dataset.loc[tweet_index, "label"]
            self.true_sentiment.config(text=true_sentiment)

    def analyze_sentiment(self):
        try:
            # Get text input from the user
            user_text = self.text_input.get()

            # Ensure the text input is not empty
            if not user_text.strip():
                messagebox.showerror("Error", "Please enter some text for analysis.")
                return

            # Tokenize the input text and make prediction
            inputs = self.tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predicted_label_id = torch.argmax(logits, dim=-1).item()

            # Define sentiment labels
            labels = ["Bearish", "Bullish", "Neutral"]
            predicted_label = labels[predicted_label_id]

            # Display the predicted sentiment
            self.predicted_sentiment.config(text=predicted_label)

            messagebox.showinfo("Prediction", f"Predicted Sentiment: {predicted_label}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze sentiment: {str(e)}")


# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisGUI(root)
    root.mainloop()
