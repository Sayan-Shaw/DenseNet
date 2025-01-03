import torch
import os
import shutil

class Checkpoint:
    @staticmethod
    def latest(opt):
        """
        Load the latest checkpoint.
        Args:
            opt: Configuration options containing 'resume' path.
        Returns:
            latest: Dictionary containing checkpoint information.
            optim_state: Optimizer state.
        """
        if opt['resume'] == 'none':
            return None, None

        latest_path = os.path.join(opt['resume'], 'latest.pth')
        if not os.path.isfile(latest_path):
            return None, None

        print(f"=> Loading checkpoint from {latest_path}")
        latest = torch.load(latest_path, map_location='cpu')
        optim_state = torch.load(os.path.join(opt['resume'], latest['optimFile']), map_location='cpu')

        return latest, optim_state

    @staticmethod
    def save(epoch, model, optim_state, is_best_model, opt):
        """
        Save model and optimizer state.
        Args:
            epoch: Current epoch number.
            model: Model to save.
            optim_state: Optimizer state to save.
            is_best_model: Boolean indicating if it's the best model so far.
            opt: Configuration options containing 'save' directory.
        """
        # Ensure save directory exists
        os.makedirs(opt['save'], exist_ok=True)

        # Save model and optimizer state
        model_file = f"model_{epoch}.pth"
        optim_file = f"optimState_{epoch}.pth"

        # Save model state dict
        torch.save(model.state_dict(), os.path.join(opt['save'], model_file))
        torch.save(optim_state, os.path.join(opt['save'], optim_file))

        # Save latest checkpoint metadata
        latest_checkpoint = {
            'epoch': epoch,
            'modelFile': model_file,
            'optimFile': optim_file,
        }
        torch.save(latest_checkpoint, os.path.join(opt['save'], 'latest.pth'))

        # Save best model separately
        if is_best_model:
            shutil.copyfile(
                os.path.join(opt['save'], model_file),
                os.path.join(opt['save'], 'model_best.pth')
            )
