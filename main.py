import argparse
from train import train_model

def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    if args.train:
        train_model(num_epochs=args.num_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)

if __name__ == '__main__':
    main()