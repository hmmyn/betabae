import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import random

from betabae.text_core import TextAgent
from betabae.text_logger import TextLogger, GitaDataset

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_text_agent(
    dataset_path,
    output_dir,
    n_epochs=50,
    batch_size=8,
    seq_len=128,
    d_model=64,
    n_layers=2,
    n_heads=4,
    lr=3e-4,
    vocab_size=256,
    log_every=10,
    generate_every=5,
    device='cpu'
):
    """Train text generation agent on Gita dataset"""
    
    # Set up directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    
    # Set seed
    set_seed(42)
    
    # Load dataset
    print("Loading Gita dataset...")
    dataset = GitaDataset(dataset_path, seq_len=seq_len, vocab_size=vocab_size)
    
    # Create agent
    print("Creating text agent...")
    agent = TextAgent(
        vocab_size=len(dataset.vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        lr=lr
    ).to(device)
    
    # Create logger
    logger = TextLogger(log_dir)
    
    print(f"Training on {len(dataset.tokens)} sequences")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Training loop
    step = 0
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Training steps per epoch
        steps_per_epoch = max(1, len(dataset.tokens) // batch_size)
        
        for step_in_epoch in range(steps_per_epoch):
            # Get batch
            batch = dataset.get_batch(batch_size).to(device)
            
            # Forward pass
            logits, attention_weights = agent(batch)
            
            # Compute loss
            loss = agent.loss(logits, batch)
            
            # Backward pass
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            
            # Log step
            logger.log_step(
                attention_weights=attention_weights,
                hidden_state=logits[:, -1, :],  # Last token hidden state
                loss=loss.item(),
                perplexity=perplexity
            )
            
            step += 1
            
            # Print progress
            if step % log_every == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}, Perplexity={perplexity:.2f}")
        
        # Generate sample text
        if epoch % generate_every == 0:
            print(f"\nGenerating sample text at epoch {epoch+1}:")
            
            # Generate from a prompt
            prompt = "The soul is eternal"
            prompt_tokens = dataset.encode_text(prompt)
            
            # Pad or truncate prompt
            if len(prompt_tokens) < seq_len:
                prompt_tokens = prompt_tokens + [0] * (seq_len - len(prompt_tokens))
            else:
                prompt_tokens = prompt_tokens[:seq_len]
            
            prompt_tensor = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
            
            # Generate
            with torch.no_grad():
                generated = agent.generate(
                    prompt_tensor,
                    max_length=100,
                    temperature=0.8,
                    top_k=50
                )
            
            # Decode and display
            generated_text = dataset.decode_tokens(generated[0])
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text[:200]}...")
            
            # Log generation
            logger.log_generation(generated_text[:200], step)
        
        # Save epoch data
        logger.save_epoch(epoch)
        
        # Print epoch summary
        avg_loss = np.mean(logger.losses) if logger.losses else 0
        avg_perplexity = np.mean(logger.perplexities) if logger.perplexities else 0
        print(f"Epoch {epoch+1} complete: Avg Loss={avg_loss:.4f}, Avg Perplexity={avg_perplexity:.2f}")
    
    # Save final model
    torch.save(agent.state_dict(), output_dir / 'final_model.pt')
    print(f"\nTraining complete! Model saved to {output_dir / 'final_model.pt'}")
    
    return agent, dataset, logger

def main():
    parser = argparse.ArgumentParser(description='Train BetaBae text generation agent on Gita')
    parser.add_argument('--dataset', required=True, help='Path to Gita CSV file')
    parser.add_argument('--output', default='./text_outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--vocab_size', type=int, default=256, help='Vocabulary size')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Train the agent
    agent, dataset, logger = train_text_agent(
        dataset_path=args.dataset,
        output_dir=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        vocab_size=args.vocab_size,
        device=args.device
    )
    
    print("\nTraining completed successfully!")
    print(f"Check outputs in: {args.output}")

if __name__ == '__main__':
    main()
