# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:51:56 2021

@author: zhuyupeng
"""

#use = 18354285409
#key_words = 285409
#key_words = 1990ab1213

# 导入函数库
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import dateutil
import datetime



'''
获取数据
ts.set_token('b3828ec638ae1eb9ddbbea99ff964d2bd8d3f4a41578d3a73cc6012f')
pro = ts.pro_api()
df = pro.trade_cal()
print(df)
df.to_csv("trade_cal.csv")
'''

CASH = 10000
Start_date ="2015-01-01"
End_date = "2015-12-01"

pro = ts.pro_api()

trade_cal = pd.read_csv("trade_cal.csv")
trade_cal["is_open"]=trade_cal["is_open"].astype('str')
trade_cal["cal_date"] = pd.to_datetime(trade_cal["cal_date"])
#print(trade_cal)

#print(type(trade_cal["cal_date"]))


#上下文
class Context:
    def __init__(self, cash, start_date, end_date):
        self.cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.position = {}
        self.benchmark = None
        self.date_range = trade_cal[(trade_cal["is_open"]==1)&
                                    (trade_cal["cal_date"]<=start_date)&
                                    (trade_cal["cal_date"]>=end_date)]["cal_date"].values
        self.dt = None #bug
                                    
class G:
    pass

g = G()


#context =   Context(1000, 20160101,20170101)     
  
                                                             
context = Context(CASH, Start_date,End_date)    
#print(context.date_range)

def set_benchmark(security):  #支持一支股票
    context.benchmark = security


                                    
def attribute_history(security,count,fields=("open",'close','high','low','volume')):
    end_date = (context.dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = trade_cal[(trade_cal["is_open"]==1)&
                           (trade_cal["cal_date"]>=end_date)][-count:].iloc[0, :]["cal_date"]                             
    #print(start_date,end_date)       
    return attribute_daterange_histtory(security,start_date,end_date,fields)


def attribute_daterange_histtory(security,start_date,end_date,fields=("open",'close','high','low','vol')):
    try:
       f = open(security+".csv","r")      
       df = pd.read_csv(f,index_col = 0,parse_dates=["trade_date"]).loc[start_date:end_date, :]
    except:
        ts.set_token('b3828ec638ae1eb9ddbbea99ff964d2bd8d3f4a41578d3a73cc6012f')
        df= pro.daily(security,start_date,end_date)
    return df[list(fields)]


#print(attribute_daterange_histtory("601318","20110101","20170101"))                    
#today = context.dt.strftime("%Y-%m-%d")
#print(context.dt)
#print(ts.get_k_data("601318",today,today))


def get_todey_data(security):
    #print(context.dt)
    #print(type(context.dt))
    today = context.dt.strftime("%Y-%m-%d")
    try:
       f = open(security+".csv","r")
       data = pd.read_csv(f,index_col=0 ,parse_dates=["trade_date"]).loc[today, :] 
    except FileNotFoundError:
        ts.set_token('b3828ec638ae1eb9ddbbea99ff964d2bd8d3f4a41578d3a73cc6012f')
        data= pro.daily(security,today,today).iloc[0,:]
    except KeyError:
        data = pd.Series()                              
    return data
  
#print(get_todey_data("601318")) 

       
def _order(today_data,security,amount):
    p = today_data["open"]
    #print(p)
    if len(today_data) == 0:
        print("今日停牌")
        return
    
    if context.cash - amount * p<0:
        amount = int(context.cash/p)
        print("现金不足，已经调整为%d" % (amount))
        
    if amount % 100 != 0:
        if amount != -context.position.get(security,0):
            amount = int(amount / 100) *100
            print("不是100的倍数，已经调整为%d" %amount)
            
    if context.position.get(security,0) < -amount:
        amount = -context.position.get(security,0)
        print("卖出股票不能超过持仓数，已经调整为%d"%amount)
        
    context.position[security] = context.position.get(security,0) + amount
    context.cash -= amount*p
    
    if context.position[security] == 0:
        del context.position[security]


#_order(get_todey_data("601318"),"601318",10000000000)
#print(context.position)

#_order(get_todey_data("601318"),"601318",-3)
#print(context.position)                             
 

def order(security,amount):
    today_data  = get_todey_data(security)
    _order(today_data, security, amount)        
            
          
def order_target(security,amount):
    if amount < 0:
        print("数据不能为负数，已经调整为零")
        amount = 0
    today_data = get_todey_data(security)
    hold_amount = context.position.get(security,0)
    delta_amount = amount - hold_amount
    _order(today_data,security,delta_amount)        
          
            
def order_value(security,value):
    today_data = get_todey_data(security)
    amount = int(value/today_data["open"])
    _order(today_data, security, amount)        
            
          
def order_target_value(security,value):
    today_data = get_todey_data(security)
    if value<0:
       print("价值不能为负，已经调整为零")
       value = 0
       
    hold_value = context.position.get(security,0) * today_data['open']           
    delta_value = value - hold_value
    order_value(security, delta_value)        
  
            
#order_target("601318",520) 
#order_target_value("600519",30000)   
#print(context.position)    
#print(context.date_range)
#print(type(context.date_range))

def run():
    plt_df = pd.DataFrame(index = pd.to_datetime(context.date_range),columns = ["value"])
    init_value = context.cash
    initialize(context)
    last_prize = {}
    for dt in context.date_range:
        dt =str(dt)
        context.dt = dateutil.parser.parse(dt)
        print(context.dt)
        handle_data(context)
        value = context.cash
        
        for stock in context.position:
            #考虑停牌
            today_data = get_todey_data(stock)
            if len(today_data) == 0:
                p = last_prize[stock]
            else:
                p = today_data['open']
                last_prize[stock] = p
            value += p*context.position[stock]
        plt_df.loc[dt,"value"] = value
    plt_df['ratio'] = (plt_df['value']-init_value)/init_value
    bm_df = attribute_daterange_histtory(context.benchmark, context.start_date, context.end_date)
    print(bm_df)
    bm_init = bm_df['open'][0]
    plt_df['benchmark_ratio'] = (bm_df['open']-bm_init) / bm_init
    #print(plt_df)
    print(plt_df)
    print(context.dt)
    plt_df[["ratio","benchmark_ratio"]].plot()
    plt.show()

def initialize(context):
    set_benchmark("000001.SZ")
    g.p1 = 5
    g.p2 = 60
    g.security="000001.SZ"

def handle_data(context):
    hist = attribute_history(g.security, g.p2)
    ma5 = hist['close'][-g.p1:].mean()
    ma60 = hist['close'].mean()
    
    if ma5 > ma60 and g.security not in context.position:
        order_value(g.security, context.cash)
    elif ma5 < ma60 and g.security in context.position:
        order_target(g.security, 0)

run()

#df = pro.daily(ts_code='000001.SZ', start_date='20100701', end_date='20180718')
    
  
    
  
    