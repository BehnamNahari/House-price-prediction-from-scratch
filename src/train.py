import numpy as np
from src.config import (RAW_DATA_DIR, TARGET_COLUMN, VAL_RATIO, RANDOM_SEED,LEARNING_RATE,N_EPOCHS,LAMBDA_REG,MODEL_PATH,
    PREPROCESSOR_PATH)
from src.data_loading import load_data,split_features_target
from src.preprocessing import train_val_split,prepare_numeric_features
from src.model import LinearRegression
from src.evaluate import rmse, r2_score
from src.utils import (create_directories, set_seed, timer)
from src.persistence import (save_model, save_preprocessor)

def main():

    create_directories()
    set_seed(RANDOM_SEED)
    start_time = timer()

    train_path = RAW_DATA_DIR / 'train.csv'
    df = load_data(train_path)
    df = df.dropna(subset=[TARGET_COLUMN])
    x,y = split_features_target(df,TARGET_COLUMN)
    y = np.log1p(y)
    x_train, x_val, y_train, y_val = train_val_split(x,y,val_ratio=VAL_RATIO,)
    x_train_np, x_val_np, numeric_cols, medians, mean, std = prepare_numeric_features(x_train,x_val)
    y_train_np = y_train.to_numpy(dtype=np.float64)
    y_val_np = y_val.to_numpy(dtype=np.float64)

    model = LinearRegression(lr=LEARNING_RATE,n_epochs=N_EPOCHS,reg_lambda=LAMBDA_REG)
    model.fit(x_train_np, y_train_np,x_val_np, y_val_np,verbose=True)
    save_model(model)
    save_preprocessor(
        numeric_cols=numeric_cols,
        medians=medians,
        mean=mean,
        std=std,
    )
    train_pred = model.predict(x_train_np)
    val_pred = model.predict(x_val_np)

    train_rmse = rmse(y_train_np,train_pred)
    val_rmse = rmse(y_val_np,val_pred)
    train_r2 = r2_score(y_train_np,train_pred,)

    val_r2 = r2_score(y_val_np,val_pred)

    print("\nFinal results")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Val RMSE: {val_rmse:.6f}")
    print(f"Train R2 : {train_r2:.4f}")
    print(f"Val R2   : {val_r2:.4f}")
    elapsed = timer() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

if __name__=='__main__':
    main()