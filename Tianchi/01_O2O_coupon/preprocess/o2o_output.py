import pickle

test = pickle.load(open('../data/test.pickle', "rb"))
predictors = ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'distance',
              'weekday', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
              'weekday_7', 'weekday_type',
              'u_coupon_count', 'u_buy_count', 'u_buy_with_coupon', 'u_merchant_count', 'u_min_distance',
              'u_max_distance', 'u_mean_distance', 'u_median_distance', 'u_use_coupon_rate', 'u_buy_with_coupon_rate',
              'm_coupon_count', 'm_sale_count', 'm_sale_with_coupon', 'm_min_distance', 'm_max_distance',
              'm_mean_distance', 'm_median_distance', 'm_coupon_use_rate', 'm_sale_with_coupon_rate', 'um_count',
              'um_buy_count',
              'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate', 'um_coupon_use_rate', 'um_buy_with_coupon_rate']

with open('../result/3_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
# test prediction for submission
y_test_pred = model.predict(test[predictors])
# print(test.shape, y_test_pred.shape)
submit = test[['User_id', 'Coupon_id', 'Date_received']].copy()
submit['Label'] = y_test_pred
submit.to_csv('../result/submit1.csv', index=False, header=False)
print(submit.head())
