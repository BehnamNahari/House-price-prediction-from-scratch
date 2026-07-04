import numpy as np
from src.config import (RAW_DATA_DIR, TARGET_COLUMN, VAL_RATIO, RANDOM_SEED,LEARNING_RATE,N_EPOCHS,LAMBDA_REG )
from src.data_loading import load_data,split_features_target
from src.preprocessing import train_val_split,prepare_numeric_features
from src.model import LinearRegression
from src.evaluate import rmse
from src.utils import (
    create_directories,
    set_seed,
    timer,
)

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
    train_pred = model.predict(x_train_np)
    val_pred = model.predict(x_val_np)

    train_rmse = rmse(y_train_np,train_pred)
    val_rmse = rmse(y_val_np,val_pred)

    print("\nFinal results")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Val RMSE: {val_rmse:.6f}")
    elapsed = timer() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

if __name__=='__main__':
    main()