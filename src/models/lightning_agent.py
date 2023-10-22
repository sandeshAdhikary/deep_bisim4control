import lightning.pytorch as pl

class LightningAgent(pl.LightningModule):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def training_step(self, batch, batch_idx):
        """
        batch is a sample from the replay buffer
        """
        obs, action,_, reward, next_obs, not_done = batch
        self.agent.update(batch)

    def configure_optimizers(self):
        pass
    
    def get_action(self, obs, batched=False):
        return self.agent.sample_action(obs, batched)
