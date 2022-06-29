import torch
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter


def train(model, dataloader, num_epochs, criterion, optimizer, logging_info):

    name_logging, name_model, name_writer, name = logging_info

    writer = SummaryWriter(log_dir=name_writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float("inf")
    niter = 1
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[0]
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, features = model(inputs)
                    optimizer.zero_grad()
                    loss = criterion(inputs, output)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.detach().item() / batch_size
                running_loss += batch_loss
                batch_num += 1
                writer.add_scalar("/Loss", batch_loss, niter)
                niter += 1
                tepoch.set_postfix(loss=batch_loss)

                if batch_num % 10 == 0:
                    logging.info(
                        f"[{epoch}/{num_epochs}]"
                        f"[{batch_num}/{len(dataloader)}]"
                        f"LossTot: {batch_loss}"
                    )

            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, name_model)
                logging.info(
                    f"Saving model to {name_model} with loss = {best_loss}."
                )
                print(f"Saving model to {name_model} with loss = {best_loss}.")
    return model, name_logging, name_model, name_writer, name
