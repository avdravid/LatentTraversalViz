def train_gen(labels):
  
    # generating imamges
    fake_conditions_latent_vec =  condition_to_latent_vec(labels)
    conditions = encodeOneHot(labels).to(device)

    z1 = torch.randn(BATCH_SIZE, nz1).to(device)

    netG.zero_grad()

    sample = netG(z1, fake_conditions_latent_vec)
    
    # get feedback from discriminator and classifier
    valid_outputs = netD(sample)
    clf_outputs,_ = Classifier(sample)
    
    # train generator
    ls = c1(valid_outputs, real_labels)  # source loss (real/fake)
    lc = c2(clf_outputs, conditions)     # class loss

    gloss = lc + ls

    gloss.backward()

    optimizerG.step()

    return loss
