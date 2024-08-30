# The code should contain user defined functions and dictionaries
# The menu system should be user-friendly. Learners are encouraged to
# think creatively to create the user menu

# The application allows users to order supermarket
# grocery items online:
# 1 view the list of products by category
# 2 place orders
# 3 view the final bill electronically using the application
#               8 items per category
import time
import random
z={'prices':{'dairy':[2.3,4.5,3.4,3.15,1.4,12.5,5.3,0.95],'packaged goods':[2.7,7,3.1,2.6,2.1,2,1.05,1.05],'canned goods':[1.45,1.15,1.35,1.45,1.25,1.1,1.95,0.95],'condiments and sauces':[0.8,1.3,3.15,2.65,4.5,3.75,3.2,4.95],'drink and beverages':[15,31,15,3.9,7,0.8,9.9,12.4]}}
dd={'categories':{'dairy':{'milk':2.3,'butter':4.5,'eggs':3.4,'cheese slices':3.15,'evaporated milk creamer':1.4,'milo':12.5,'biscuits':5.3,'yoghurt':0.95},'packaged goods':{'bread':2.7,'cereal':7,'crackers':3.1,'chips':2.6,'raisin':2.1,'nuts':2,'green bean':1.05,'barley':1.05},'canned goods':{'tomato':1.45,'button mushroom':1.15,'baking bean':1.35,'tuna fish':1.45,'kernel corn':1.25,'sardine fish':1.1,'chicken luncheon meat':1.95,'pickled lettuce':0.95},'condiments/sauces':{'fine salt':0.80,'sea salt flakes':1.3,'chicken stock':3.15,'chilli sauce':2.65,'oyster sauce':4.5,'sweet soy sauce':3.75,'tomato ketchup':3.2,'sesame oil':4.95},'drinks and beverages':{'green tea cans':15,'blackcurrent ribena':31,'100 plus cans':15,'orange cordial':3.9,'bottled mineral water':7,'pineapple juice':0.80,'nescafe coffee':9.90,'coke cans':12.40}}}
x={'categories':{'dairy':{'items':['milk','butter','eggs','cheese slices','evaporated milk creamer','milo','biscuits','yoghurt']},'packaged goods':{'items':['bread','cereal','crackers','chips','raisin','nuts','green bean','barley']},'canned goods':{'items':['tomato','button mushroom','baking bean','tuna fish','kernel corn','sardine fish','chicken luncheon meat','pickled lettuce']},'condiments/sauces':{'items':['fine salt','sea salt flakes','chicken stock','chilli sauce','oyster sauce','sweet soy sauce','tomato ketchup','sesame oil']},'drinks and beverages':{'items':['green tea cans','blackcurrent ribena','100 plus cans','orange cordial','mineral water','pineapple juice','nescafe coffee','coke cans']}}}
y={'options':['view complete selection of products','buy groceries','view your shopping cart','exit from mart']}
cat={'categories':['all','dairy products', 'packaged goods', 'canned goods','sauces/condiments','drinks and beverages']}
shop_cart={'product':[],'base amount':[], 'quantity':[], 'amount':[]}
cats=['dairy', 'packaged goods', 'canned goods','sauces/condiments','drinks and beverages']
def main(): #header
    u='*'*15
    print('{:<11s}{:^3s}'.format('', 'welcome to SHOP\'N\'SAVE online supermarket'))
    print('{} number one online supermarket {}'.format(u,u))
def price(): #to sort by either ascending or descending price
    global nm
    nm=0
    if k==2:
        nm= str(input('would you like to sort by ascending or descending? (a/d)    '))
        if nm!= 'a' and nm!='d':
            print('please enter a valid option')
            price()
        brr()
def alpha():
    global al
    if k==3:
        al=str(input('would you like to sort by ascending or descending? (a/d)    '))
        if al!= 'a' and al!='d':
            print('please enter a valid option')
            alpha()
        brr()
def menui():
    global sk
    st()
    print('{:^15.0f}> {:^3s}'.format(1, y['options'][0]))
    st()
    kk()
    if k >= 1 and k <= 4:
        if k == 1:
            print('you have chosen to {}'.format(y['options'][k - 1]))
            if k == 1:  # option to view in category
                sk = 1
                opt1()
            if k == 2:
                sk += 1
                price()
            if k == 3:
                sk += 2
                alpha()
def menu(): #to be printed after the user input choice in menu, to notify user of their choice
    global sk
    st()
    for i in range(4):
        print('{:^15.0f}> {:^3s}'.format(i+1,y['options'][i]))
    st()
    kk()
    if k>=1 and k<=4:
        if k==1:
            print('you have chosen to {}'.format(y['options'][k-1]))
            if k == 1:  # option to view in category
                sk=1
                opt1()
            if k==2:
                sk+=1
                price()
            if k==3:
                sk+=2
                alpha()
        if k==2: #buy groceries
            yyr = 4
            sk=0
            buymenu3()
            buygroceries()
        if k==3:
            viewcart()
    #     opt 4 is done
    if k > 4:
        print('please enter a valid option')
        time.sleep(2)
        menu()
def menu2():
    sk=1
    if k >= 1 and k <= 4:
        if k == 1:  # option to view in category
            sk = 1
            opt1()
        if k == 2:
            sk += 1
            price()
        if k == 3:
            sk += 2
            alpha()
    else:
        print('please enter a valid option')
def st(): #segregation
    p = '-' * 12
    print(p * 5)
def kk(): #enter choice
    global k
    try:
        k = int(input('enter your choice here         '))
        if k <= 0:
            kk()
    except ValueError:
        print('')
        quit('please enter a valid integer')
def buygroceriesmenu():
    print('you have chosen to {}'.format(y['options'][k - 1]))
    print('would you like to 1) continue buying groceries')
    print('{:>20s}{:^7s}'.format('2)', ' back to previous menu'))
    iii = (input('enter your choice             '))
    if iii == '1':
        buymenu3()
    if iii == '2':
        menu()
    if iii != '1' and iii != '2':
        print('****** please enter a valid choice ******')
        buygroceriesmenu()
def buygroceriesmenuend():
    print('would you like to 1) view your cart')
    print('{:>20s}{:^7s}'.format('2)', ' checkout'))
    print('{:>20s}{:^7s}'.format('3)', ' back to previous menu'))
    iii = (input('enter your choice             '))
    if iii == '1':
        viewcart()
        viewcartend()
    if iii == '2':
        checkout()
    if iii == '3':
        menu()
    else:
        print('****** please enter a valid choice ******')
        buygroceriesmenuend()
def buymenu3():
    global iii
    print('would you like to ')
    print('{:>20s}{:^7s}'.format('1)', ' add to cart'))
    print('{:>20s}{:^7s}'.format('2)', ' checkout '))
    iii = (input('enter your choice             '))
    if iii == '1':
        buygroceries()
        buygroceriesmenu()
    if iii == '2':
        checkout()
    if iii != '1' and iii != '2':
        print('****** please enter a valid choice ******')
        buymenu3()
def u(): #this is for after each option; to select quit, cont, etc
    print('would you like to 1) continue shopping')
    print('{:>20s}{:^7s}'.format('2)', 'quit'))
    pv=(input('enter your choice             '))
    if pv=='1':
        menu()
    if pv=='2':
        quit("thank you for shopping online with SHOP\'N\'SAVE, have a good day!")
    if pv!='1' and pv!='2':
        print('****** please enter a valid choice ******')
        u()
def viewcart():
    vvvv=len(shop_cart['product'])
    if vvvv==0:
        st()
        time.sleep(2)
        print('currently cart is empty...')
        print('please add items into cart first')
        time.sleep(1)
        st()
        print('would you like to ')
        print('{:>20s}{:^7s}'.format('1)', ' add to cart'))
        print('{:>20s}{:^7s}'.format('2)', ' view categories '))
        print('{:>20s}{:^7s}'.format('3)', ' to main menu '))
        while True:
            iii = (input('enter your choice             '))
            if iii == '1':
                buygroceries()
                buygroceriesmenu()
            if iii=='2':
                menui()
            if iii=='3':
                menu()
            if iii != '1' or  iii != '2' or iii != '3':
                print('****** please enter a valid choice ******')
            break
    else:
        print('you are currently viewing your cart')
        st()
        print('** DISCLAIMER! prices displayed are inclusive of GST **')
        print('{:>29s}'.format('and'))
        print('** DISCLAIMER! price($) indicates individual prices **')
        time.sleep(2)
        st()
        print('{}\t\t{}\t\t\t{}\t\t\t{}'.format('no.','name','quantity','price($)'))
        for i in range(len(shop_cart['product'])):
            basegst = shop_cart['base amount'][i]
            totandgst = sum(shop_cart['amount'])
            print('{}{:^18s}{:>10.0f}{:.17s}{:>20.2f}'.format(i + 1, shop_cart['product'][i], shop_cart['quantity'][i], '×',
                                                              (basegst / 100) * 107))
        st()
        print('{}{:^9.0f}{:>14.0f}{:.15s}{:>20.2f}'.format('total:', len(shop_cart['product']), sum(shop_cart['quantity']),
                                                           '×', (totandgst / 100) * 107))
        st()
def uu():
    print('would you like to 1) continue viewing catalogues?')
    print('{:>20s}{:^7s}'.format('2)', ' view main menu?'))
    pv = int(input('enter your choice             '))
    if pv == 1:
        opt1()
    if pv == 2:
        menu()
    if pv > 2:
        print('****** please enter a valid choice ******')
        u()
def r(): #menu for viewing in category
    global xxv
    xxv = cat.get('categories')
    print('we are always stocked up with wide range of products to meet all your needs!')
    print('choose a category by number to view it\'s catalogue')
    print('** DISCLAIMER! prices displayed are not inclusive of GST **')
    st()
    for kt in range(6):
        print('{:^15.0f}> {:^3s}'.format(kt+1,xxv[kt]))
    st()
    kk()
    jjk()
    if sk==2:
        if nm=='a' and 'd':
            while True:
                kk()
                jjk()
    if sk==3:
        if nm=='a' and 'd':
            while True:
                kk()
                jjk()
def frfr():
    global bbb
    global yyr
    yyr = 0
    while True:
        st()
        bbb = str(input('enter category you\'d like to purchase from: '))
        if bbb == 'dairy' or bbb == 'packaged goods' or bbb == 'canned goods' or bbb == 'condiments/sauces' or bbb == 'drinks and beverages':
            yyr += 4
            allcatalogue2()
            while True:
                ccc = str(input('enter product name: '))
                xyy = dd['categories'].get(bbb)
                if ccc in xyy:
                    pricekey = xyy[ccc]
                    while True:
                        try:
                            ddd = int(input('enter the quantity of {}:  '.format(ccc.upper())))
                            if ddd >= 1:
                                ux = ((pricekey * ddd) / 100) * 107
                                print(
                                    '>you have added {}× {} into cart. \n>total ${:.2f}, inclusive of 7% GST'.format(
                                        ddd,
                                        ccc.upper(),
                                        ux))
                                print('\neach {} is priced at ${:.2f}, excluding 7% GST'.format(ccc.upper(), pricekey))
                                st()
                                if ccc in shop_cart['product']:
                                    i = shop_cart['product'].index(ccc)
                                    shop_cart['base amount'][i] = pricekey
                                    shop_cart['product'][i] = ccc
                                    shop_cart['quantity'][i] = ddd
                                    shop_cart['amount'][i] = pricekey * ddd
                                    print('overwriting the previous the quantity entered...')
                                    time.sleep(2)
                                    print('there is now {}× {} in cart'.format(ddd, ccc.upper()))
                                else:
                                    shop_cart['base amount'].append(pricekey)
                                    shop_cart['product'].append(ccc)
                                    shop_cart['quantity'].append(ddd)
                                    shop_cart['amount'].append(pricekey * ddd)
                                time.sleep(2)
                                userintt = str(input('would you like to continue adding into cart? y/n   '))
                                if userintt == 'y':
                                    frfr()
                                    if userintt == 'n':
                                        break
                                    break
                                if userintt == 'n':
                                    break
                            if ddd < 1:
                                print('please enter a valid number; must be one, or more than one')
                        except ValueError:
                            print('please enter an integer!')
                    break
                else:
                    print('please enter a valid product')
            break
        else:
            print('please enter a valid category')
    st()
    buygroceriesmenuend()
def buygroceries():
    global bbb
    if iii == '2':
        print('you have chosen to {}'.format('checkout'))
    if iii == '1':
        st()
        print('{:>37s}'.format('** DISCLAIMER! **'))
        print('\t     ** entering the same product twice will **\n\t        ** overwrite the previous emtry **')
        st()
        time.sleep(4)
        yyr = 0
        sk = 0
        print('our categories')
        st()
        for key in x['categories']:
            print('{:^59s}'.format(key))
        while True:
            st()
            bbb = str(input('enter category you\'d like to purchase from: '))
            if bbb == 'dairy' or bbb == 'packaged goods' or bbb == 'canned goods' or bbb == 'condiments/sauces' or bbb == 'drinks and beverages':
                yyr += 4
                allcatalogue2()
                while True:
                    ccc = str(input('enter product name: '))
                    xyy = dd['categories'].get(bbb)
                    if ccc in xyy:
                        pricekey = xyy[ccc]
                        while True:
                            try:
                                ddd = int(input('enter the quantity of {}:  '.format(ccc.upper())))
                                if ddd >= 1:
                                    ux = ((pricekey * ddd) / 100) * 107
                                    print(
                                        '>you have added {}× {} into cart. \n>total ${:.2f}, inclusive of 7% GST'.format(
                                            ddd,
                                            ccc.upper(),
                                            ux))
                                    print('\neach {} is priced at ${:.2f}, excluding 7% GST'.format(ccc.upper(),
                                                                                                    pricekey))
                                    st()
                                    if ccc in shop_cart['product']:
                                        i = shop_cart['product'].index(ccc)
                                        shop_cart['base amount'][i] = pricekey
                                        shop_cart['product'][i] = ccc
                                        shop_cart['quantity'][i] = ddd
                                        shop_cart['amount'][i] = pricekey * ddd
                                        print('overwriting the previous the quantity entered...')
                                        time.sleep(2)
                                        print('there is now {}× {} in cart'.format(ddd, ccc.upper()))
                                    else:
                                        shop_cart['base amount'].append(pricekey)
                                        shop_cart['product'].append(ccc)
                                        shop_cart['quantity'].append(ddd)
                                        shop_cart['amount'].append(pricekey * ddd)
                                    time.sleep(2)
                                    userintt = str(input('would you like to continue adding into cart? y/n   '))
                                    if userintt == 'y':
                                        frfr()
                                        if userintt == 'n':
                                            break
                                        break
                                    if userintt == 'n':
                                        break
                                if ddd < 1:
                                    print('please enter a valid number; must be one, or more than one')
                            except ValueError:
                                print('please enter an integer!')
                        break
                    else:
                        print('please enter a valid product')
                break
            else:
                print('please enter a valid category')
        st()
        buygroceriesmenuend()
def remove():
    print('{:^25s}'.format('which items would you like to remove?'))
    print('** DISCLAIMER! **\n** removing a product will remove all of its contents out! **')
    time.sleep(3)
    removing=str(input('enter name of product     '))
    if removing in shop_cart['product']:
        v=shop_cart['product']
        i=v.index(removing)
        del (v[i])
        del (shop_cart['base amount'][i])
        del (shop_cart['amount'][i])
        del (shop_cart['quantity'][i])
        viewcart()
    if removing not in shop_cart['product']:
        print('...')
        remove()
def checkout():
    global discount
    print('{:^60s}'.format('\n\t\t\t\t**** welcome to checkout! ****'))
    st()
    print('fetching your shopping cart...')
    time.sleep(4)
    viewcart()
    totalwgst = ((sum(shop_cart['amount'])) / 100) * 107
    discprice=(totalwgst/100)*90
    print('members enjoy 10% OFF their entire cart!\nhead over to SHOPNSAVEMEMBERS.com to sign up today!\nT&Cs apply')
    st()
    discount=(input('do you have a membership card? y/n   '))
    if discount=='y':
        print('now calculating your total payable amount...')
        time.sleep(1.5)
        print('your total amount payable inclusive of 10% discount is ${:.2f} '.format(discprice))
        st()
        payment()
    if discount=='n':
        print('now calculating your total payable amount...')
        time.sleep(1.5)
        print('your total amount payable is ${:.2f} '.format(totalwgst))
        st()
        payment()
    if discount!='y' and discount!='n':
        print('please enter valid input')
        checkout()
def payment():
    print('\nwelcome to the ePayment page,\nwe accept 1. VISA, 2. MASTERCARD and 3. PayPal.')
    while True:
        card = str(
            input('\nplease enter your preferred payment type (1/2/3)     '))
        if card == '1' or card == '2' or card == '3':
            print('to proceed, please enter the relevant card details,\n ')
            if card == '1':
                card = 'VISA'
            if card == '2':
                card = 'MASTERCARD'
            if card == '3':
                card = 'PayPal'
            while True:
                cardno = str(input('please enter the 8 digits on your card, without any spaces\n> '))
                if len(cardno) == 8:
                    while True:
                        lastthree = str(input('\nplease enter the 3 digit security code\n> '))
                        if len(lastthree) == 3:
                            print('you have entered a {} card,\n\tcard number {}\n\tand security number {}'.format(card,cardno,lastthree))
                            while True:
                                confirmation = str(input(
                                    '\nplease confirm that this is the right card,\nCONTINUE with payment? y/n    '))
                                if confirmation == 'y':
                                    print('processing payment')
                                    time.sleep(2)
                                    print('PAYMENT SUCCESSFUL')
                                    time.sleep(1.5)
                                    print('hold on a moment...')
                                    time.sleep(1.5)
                                    print('tabulating your receipt...')
                                    time.sleep(1)
                                    totalwgst = ((sum(shop_cart['amount'])) / 100) * 107
                                    totandgst = sum(shop_cart['amount'])
                                    discprice = (totalwgst / 100) * 90
                                    discpricewithgst=(discprice/100)*107
                                    if discount=='y':
                                        xyz = '*' * 70
                                        print('\n', xyz)
                                        print('\t\t\t\t  SHOP\'N\'SAVE ONLINE SUPER MARKET')
                                        print('\t\t\t\tyour trusted and go-to supermarket!\n')
                                        print('tel 93027 3382')
                                        print(time.strftime("%m/%d/%Y, %H:%M:%S"))
                                        print(xyz)
                                        print('receipt no.', random.randrange(100, 1000, 3))
                                        print(xyz)
                                        print('DESCRIPTIONS')
                                        print('{}\t\t\t{}\t\t\t\t{}\t\t\t\t{}'.format('no.', 'name', 'quantity',
                                                                                      'price($)'))
                                        for i in range(len(shop_cart['product'])):
                                            basegst = shop_cart['base amount'][i]
                                            print('{} {:^26s}{:>10.0f}{:.17s}{:>24.2f}'.format(i + 1,
                                                                                               shop_cart['product'][i],
                                                                                               shop_cart['quantity'][i],
                                                                                               '×',
                                                                                               (basegst / 100) * 107))
                                        print(xyz)
                                        print('discount applied @ 10% OFF {:>36.2f}'.format(totandgst - discprice))
                                        print('inclusive of GST @ 7% {:>41.2f}'.format(discpricewithgst - discprice))
                                        print('{}{:^11.0f}items{:>13.0f}{:.19s}{:>24.2f}'.format('subtotal:', len(
                                            shop_cart['product']), sum(shop_cart['quantity']),'×', discpricewithgst))
                                        print(xyz)
                                    if discount=='n':
                                        xyz = '*' * 70
                                        print('\n',xyz)
                                        print('\t\t\t\t  SHOP\'N\'SAVE ONLINE SUPER MARKET')
                                        print('\t\t\t\tyour trusted and go-to supermarket!\n')
                                        print('tel 93027 3382')
                                        print(time.strftime("%m/%d/%Y, %H:%M:%S"))
                                        print(xyz)
                                        print('receipt no.', random.randrange(100, 1000, 3))
                                        print(xyz)
                                        print('DESCRIPTIONS')
                                        print('{}\t\t\t{}\t\t\t\t{}\t\t\t\t{}'.format('no.', 'name', 'quantity',
                                                                                      'price($)'))
                                        for i in range(len(shop_cart['product'])):
                                            basegst = shop_cart['base amount'][i]
                                            print('{} {:^26s}{:>10.0f}{:.17s}{:>24.2f}'.format(i + 1,
                                                                                               shop_cart['product'][i],
                                                                                               shop_cart['quantity'][i],
                                                                                               '×',
                                                                                               (basegst / 100) * 107))
                                        print(xyz)
                                        print('inclusive of GST @ 7% {:>41.2f}'.format(totalwgst - sum(shop_cart['amount'])))
                                        print('{}{:^11.0f}items{:>13.0f}{:.19s}{:>24.2f}'.format('subtotal:', len(
                                            shop_cart['product']), sum(shop_cart['quantity']),'×', totalwgst))
                                        print(xyz)
                                    #here print the reciept
                                    print('confirmation receipt has been sent to your email!')
                                    time.sleep(3)
                                    while True:
                                        ppp = str(input('\nwould you like us to add you into our online newletter?\n\nbenefits of joining include:\nLOTS of free e-Vouchers = LOTS of savings,\nHELPFUL tips and tricks provided by professionals,\nand early insights on future promotions!\ny/n?     > '))
                                        if ppp == 'y':
                                            print('thank you for your interest! we\'ll be adding you in shortly...')
                                            break
                                        if ppp == 'n':
                                            break
                                        if ppp!='y' or "n":
                                            print('****** please enter a valid choice ******')
                                            continue
                                    quit('thank you valuable customer for shopping with SHOP\'N\'SAVE online supermart!\nwe hope to see you again soon!')
                                    break
                                if confirmation == 'n':
                                    while True:
                                        print('would you like to 1) try again')
                                        print('{:>20s}{:^7s}'.format('2)', ' view main menu'))
                                        pv = str(input('enter your choice             '))
                                        if pv == '1':
                                            payment()
                                        if pv == '2':
                                            menu()
                                        if pv!='1' or "2":
                                            print('****** please enter a valid choice ******')
                                            continue
                                    break
                            break
                        else:
                            continue
                    break
                if len(cardno) != 9:
                    continue
        else:
            continue
        break
def viewcartend():
    print('would you like to ')
    print('{:>20s}{:^7s}'.format('1)', ' checkout'))
    print('{:>20s}{:^7s}'.format('2)', ' remove groceries from cart'))
    print('{:>20s}{:^7s}'.format('3)', ' buy more groceries '))
    print('{:>20s}{:^7s}'.format('4)', ' to main menu '))
    zzzzz = (input('enter your choice             '))
    if zzzzz == '3':
        buygroceries()
        buygroceriesmenu()
    if zzzzz=='2':
        viewcart()
        remove()
    if zzzzz=='1':
        checkout()
    if zzzzz=='1':
        menu()
    else:
        print('****** please enter a valid choice ******')
        viewcartend()
def jjk():
    if k==1:
        print('displaying catalogue of all products...')
        time.sleep(2)
        st()
        all_catalogue()
        fff()
    if k==2:
        print('displaying catalogue of dairy products...')
        time.sleep(2)
        print('')
        dairy_catalogue()
        fff()
    if k==3:
        print('displaying catalogue of packaged goods...')
        time.sleep(2)
        print('')
        packaged_catalogue()
        fff()
    if k==4:
        print('displaying catalogue of canned goods...')
        time.sleep(2)
        print('')
        canned_catalogue()
        fff()
    if k==5:
        print('displaying catalogue of sauces and condiments...')
        time.sleep(2)
        print('')
        condiments_catalogue()
        fff()
    if k==6:
        print('displaying catalogue of drink and beverages...')
        time.sleep(2)
        print('')
        drink_catalogue()
        fff()
    else:
        print('please input a valid option')
        u()
def all_catalogue():
    dairy_catalogue()
    time.sleep(3)
    st()
    packaged_catalogue()
    time.sleep(3)
    st()
    canned_catalogue()
    time.sleep(3)
    st()
    condiments_catalogue()
    time.sleep(3)
    st()
    drink_catalogue()
    time.sleep(3)
    st()
def allcatalogue2():
    yyr=0
    yyr+=4
    xyz = '*' * 63
    if bbb == 'dairy':
        bb = z['prices'].get('dairy')
        vv = x['categories'].get('dairy')
        c = vv['items']
        print('selection of: dairy products | moo')
        print(xyz)
        print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices inclusive of 7% GST', '($)'))
        print(xyz)
        for iii in range(8):
            uu = bb[iii]
            print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', (uu / 100) * 107))
        print(xyz)
    if bbb == 'canned goods':
        bb = z['prices'].get('canned goods')
        vv = x['categories'].get('canned goods')
        c = vv['items']
        print('selection of: canned goods | convenience any time')
        print(xyz)
        print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices inclusive of 7% GST', '($)'))
        print(xyz)
        for iii in range(8):
            uu = bb[iii]
            print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', (uu / 100) * 107))
        print(xyz)
    if bbb=='packaged goods':
        bb = z['prices'].get('packaged goods')
        vv = x['categories'].get('packaged goods')
        c = vv['items']
        print('selection of: packaged goods | YUMMY!')
        print(xyz)
        print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices inclusive of 7% GST', '($)'))
        print(xyz)
        for iii in range(8):
            uu = bb[iii]
            print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', (uu / 100) * 107))
        print(xyz)
    if bbb == 'condiments/sauces':
        bb = z['prices'].get('condiments and sauces')
        vv = x['categories'].get('condiments/sauces')
        c = vv['items']
        print('selection of condiments and sauces | foods taste better than ever')
        print(xyz)
        print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices inclusive of 7% GST', '($)'))
        print(xyz)
        for iii in range(8):
            uu = bb[iii]
            print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', (uu / 100) * 107))
        print(xyz)
    if bbb == 'drinks and beverages':
        bb = z['prices'].get('drink and beverages')
        vv = x['categories'].get('drinks and beverages')
        c = vv['items']
        print('selection of drinks and beverages | quench your thirst')
        print(xyz)
        print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices inclusive of 7% GST', '($)'))
        print(xyz)
        for iii in range(8):
            uu = bb[iii]
            print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', (uu / 100) * 107))
        print(xyz)
def dairy_catalogue():
    if sk==1:
        while True:
            bb = z['prices'].get('dairy')
            vv = x['categories'].get('dairy')
            c = vv['items']
            xyz = '*' * 63
            if k == 1:
                print('selection of: dairy products | moo')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for iii in range(8):
                    print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
                print(xyz)
            if k == 2:
                print('selection of dairy products')
                print('>\thow did the farmer count his cows?')
                print('>\t\t\t\t\tusing a COWculator')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for iii in range(8):
                    print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
                print(xyz)
            break
    if sk==2:
        xzz = dd['categories'].get('dairy')
        xyz = '*' * 63
        if nm=='a': #ascending price
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=False)
            if k == 1:
                print('selection of: dairy products | moo')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 2:
                print('selection of dairy products')
                print('>\thow did the farmer count his cows?')
                print('>\t\t\t\t\tusing a COWculator')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if nm == 'd':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=True)
            if k == 1:
                print('selection of: dairy products | moo')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 2:
                print('selection of dairy products')
                print('>\thow did the farmer count his cows?')
                print('>\t\t\t\t\tusing a COWculator')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    if sk==3:
        xzz = dd['categories'].get('dairy')
        xyz = '*' * 63
        if al == 'a':  # ascending price
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=False)
            if k == 1:
                print('selection of: dairy products | moo')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 2:
                print('selection of dairy products')
                print('>\thow did the farmer count his cows?')
                print('>\t\t\t\t\tusing a COWculator')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if al == 'd':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=True)
            if k == 1:
                print('selection of: dairy products | moo')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 2:
                print('selection of dairy products')
                print('>\thow did the farmer count his cows?')
                print('>\t\t\t\t\tusing a COWculator')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    else:
        pass
def packaged_catalogue():
    if sk==1:
        bb = z['prices'].get('packaged goods')
        vv = x['categories'].get('packaged goods')
        c = vv['items']
        xyz = '*' * 63
        if k==1:
            print('selection of: packaged goods | YUMMY!')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
        if k==3:
            print('selection of packaged goods')
            print('>\twhat does the hot dog call his wife?')
            print('>\t\t\t\t\thoney bun')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
    if sk==2:
        xzz = dd['categories'].get('packaged goods')
        xyz = '*' * 63
        if nm == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=False)
            if k == 1:
                print('selection of: packaged goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 3:
                print('selection of packaged goods')
                print('>\twhat does the hot dog call his wife?')
                print('>\t\t\t\t\thoney bun')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if nm=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=True)
            if k == 1:
                print('selection of: packaged goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 3:
                print('selection of packaged goods')
                print('>\twhat does the hot dog call his wife?')
                print('>\t\t\t\t\thoney bun')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    if sk == 3:
        xzz = dd['categories'].get('packaged goods')
        xyz = '*' * 63
        if al == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=False)
            if k == 1:
                print('selection of: packaged goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 3:
                print('selection of packaged goods')
                print('>\twhat does the hot dog call his wife?')
                print('>\t\t\t\t\thoney bun')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if al == 'd':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=True)
            if k == 1:
                print('selection of: packaged goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 3:
                print('selection of packaged goods')
                print('>\twhat does the hot dog call his wife?')
                print('>\t\t\t\t\thoney bun')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    else:
        pass
def canned_catalogue():
    if sk==1:
        bb = z['prices'].get('canned goods')
        vv = x['categories'].get('canned goods')
        c = vv['items']
        xyz = '*' * 63
        if k==1:
            print('selection of: canned goods | convenience any time')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
        if k==4:
            print('selection of canned goods')
            print('>\thow much room should you give fungi to grow??')
            print('>\t\t\t\t\tas mushroom as possible')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
    if sk==2:
        xzz = dd['categories'].get('canned goods')
        xyz = '*' * 63
        if nm == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=False)
            if k == 1:
                print('selection of: canned goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 4:
                print('selection of canned goods')
                print('>\thow much room should you give fungi to grow??')
                print('>\t\t\t\t\tas mushroom as possible')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if nm=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=True)
            if k == 1:
                print('selection of: canned goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 4:
                print('selection of canned goods')
                print('>\thow much room should you give fungi to grow??')
                print('>\t\t\t\t\tas mushroom as possible')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    if sk==3:
        xzz = dd['categories'].get('canned goods')
        xyz = '*' * 63
        if al == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=False)
            if k == 1:
                print('selection of: canned goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 4:
                print('selection of canned goods')
                print('>\thow much room should you give fungi to grow??')
                print('>\t\t\t\t\tas mushroom as possible')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if al=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=True)
            if k == 1:
                print('selection of: canned goods | YUMMY')
                st()
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 4:
                print('selection of canned goods')
                print('>\thow much room should you give fungi to grow??')
                print('>\t\t\t\t\tas mushroom as possible')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    else:
        pass
def condiments_catalogue():
    if sk==1:
        bb = z['prices'].get('condiments and sauces')
        vv = x['categories'].get('condiments/sauces')
        c = vv['items']
        xyz = '*' * 63
        if k==1:
            print('selection of condiments and sauces | foods taste better than ever')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
        if k==5:
            xyz = '*' * 63
            print('selection of condiments and sauces')
            print('>\twhat did the hot dog say when his friend passed him in the race??')
            print('>\t\t\t\t\ti relish the fact that you\'ve mustard the strength to ketchup to me')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
    if sk==2:
        xzz = dd['categories'].get('condiments/sauces')
        xyz = '*' * 63
        if nm == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=False)
            if k == 1:
                print('selection of: | foods taste better than ever')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 5:
                print('selection of condiments and sauces')
                print('>\twhat did the hot dog say when his friend passed him in the race??')
                print('>\t\t\t\t\ti relish the fact that you\'ve mustard the strength to ketchup to me')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                    print(xyz)
        if nm=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=True)
            if k == 1:
                print('selection of: condiments and sauces | foods taste better than ever')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 5:
                print('selection of condiments and sauces')
                print('>\twhat did the hot dog say when his friend passed him in the race??')
                print('>\t\t\t\t\ti relish the fact that you\'ve mustard the strength to ketchup to me')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                    print(xyz)
    if sk==3:
        xzz = dd['categories'].get('condiments/sauces')
        xyz = '*' * 63
        if al == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=False)
            if k == 1:
                print('selection of: | foods taste better than ever')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 5:
                print('selection of condiments and sauces')
                print('>\twhat did the hot dog say when his friend passed him in the race??')
                print('>\t\t\t\t\ti relish the fact that you\'ve mustard the strength to ketchup to me')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                    print(xyz)
        if al=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=True)
            if k == 1:
                print('selection of: condiments and sauces | foods taste better than ever')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 5:
                print('selection of condiments and sauces')
                print('>\twhat did the hot dog say when his friend passed him in the race??')
                print('>\t\t\t\t\ti relish the fact that you\'ve mustard the strength to ketchup to me')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                    print(xyz)
        else:
            pass
def drink_catalogue():
    if sk==1:
        bb = z['prices'].get('drink and beverages')
        vv = x['categories'].get('drinks and beverages')
        c = vv['items']
        xyz = '*' * 63
        if k==1:
            print('selection of drinks and beverages | quench your thirst')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
        if k==6:
            xyz = '*' * 63
            print('selection of drinks and beverages')
            print('>\twhat\'s a tree\'s favourite root?')
            print('>\t\t\t\t\troot beer')
            print(xyz)
            print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
            print(xyz)
            for iii in range(8):
                print('|\t{:25s}{:>13s}{:>10.2f}|'.format(c[iii], '|$', bb[iii]))
            print(xyz)
    if sk==2:
        xzz = dd['categories'].get('drinks and beverages')
        xyz = '*' * 63
        if nm == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=False)
            if k == 1:
                print('selection of: drinks and beverages | quench your thirst ')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 6:
                print('selection of drinks and beverages')
                print('>\twhat\'s a tree\'s favourite root?')
                print('>\t\t\t\t\troot beer')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if nm=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[1], reverse=True)
            if k == 1:
                print('selection of: drinks and beverages | quench your thirst ')

                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 6:
                print('selection of drinks and beverages')
                print('>\twhat\'s a tree\'s favourite root?')
                print('>\t\t\t\t\troot beer')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
    if sk==3:
        xzz = dd['categories'].get('drinks and beverages')
        xyz = '*' * 63
        if al == 'a':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=False)
            if k == 1:
                print('selection of: drinks and beverages | quench your thirst ')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 6:
                print('selection of drinks and beverages')
                print('>\twhat\'s a tree\'s favourite root?')
                print('>\t\t\t\t\troot beer')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        if al=='d':
            sort_orders = sorted(xzz.items(), key=lambda x: x[0], reverse=True)
            if k == 1:
                print('selection of: drinks and beverages | quench your thirst ')
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
            if k == 6:
                print('selection of drinks and beverages')
                print('>\twhat\'s a tree\'s favourite root?')
                print('>\t\t\t\t\troot beer')
                print(xyz)
                print('{:25s}\t{:^13s}{:>4s}'.format('item name:', 'prices not inclusive of 7% GST', '($)'))
                print(xyz)
                for i in sort_orders:
                    print('|{:25s}{:>13s}{:>10.2f}'.format(i[0], '$', i[1]))
                print(xyz)
        else:
            pass
def fff():
    global gg
    while True:
        gg=str(input('would you like to continue browsing? (y/n)  '))
        if gg=='y':
            opt1()
            menu2()
        if gg=='n':
            u()
        else:
            print('please input a valid (y/n)')
def opt1(): #option one in main menu
    print('how would you like to {} by: '.format(y['options'][0]))
    v=['category','price','alphabetical order']
    for l in range(3):
        print('{:^15.0f}> {:^3s}'.format(l+1,v[l]))
    kk()
    if k>3 or k<1:
        while True:
            kk()
            if k<3 and k>=1:
                break
    print('you have chosen to filter by {}'.format(v[k-1]))
    st()

def bn(): #option four in main menu
    if k==4:
        st()
        quit("thank you for shopping online with SHOP\'N\'SAVE, have a good day!")
def sr(): #to be run after the option in the main menu has been input
    if k == 1: #view complete selection of products
        while True:
            brr()
            fff()
    if k==2: #buy groceries
        print('')
def brr():
    st()
    r()
#*********************************** main code to run *******************************************************************
main()
menu()
while True:
     sr()
     bn()