def train_discr(images, labels):

    # collecting images
    real_images = images.to(device)
    
    real_conditions = encodeOneHot(labels).to(device)
    
    fake_conditions_unencoded = torch.randint(0, 2, (BATCH_SIZE, 1))
    
    fake_conditions = encodeOneHot(fake_conditions_unencoded).to(device)
   
    fake_conditions_latent_vec =  condition_to_latent_vec(fake_conditions_unencoded)

    fake_images = netG(torch.randn(BATCH_SIZE, nz1).to(device), fake_conditions_latent_vec)
   
  
    # training discriminator
    optimizerD.zero_grad()

    real_valid = netD(real_images)
    fake_valid = netD(fake_images)
    
    d_loss = c1(real_valid, real_labels) + c1(fake_valid, fake_labels)

    d_loss.backward()

    optimizerD.step()

    return d_loss
