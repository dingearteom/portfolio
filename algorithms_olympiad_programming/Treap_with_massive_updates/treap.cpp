#include <iostream>
#include <bits/stdc++.h>

using namespace std;

struct item
{
    int x, y, size_, sum_;
    bool rev;
    item *l, *r;
    item(){};
    item(int x1){
        x = x1;
        y = rand()%(int(1e9));
        size_ = 1;
        sum_ = x;
        rev = false;
        l = nullptr;
        r = nullptr;
    };
};

typedef item* pitem;

pitem root = nullptr;

int size_(pitem v)
{
    return (v ? v->size_ : 0);
}

int sum_(pitem v)
{
    return (v ? v->sum_ : 0);
}

void update(pitem v)
{
    if (!v)
        return;
    v->sum_ = sum_(v->l) + sum_(v->r) + v->x;
    v->size_ = size_(v->l) + size_(v->r) + 1;
}

void push(pitem v)
{
    if (!v)
        return;
    if (v->rev)
    {
        v->rev = false;
        swap(v->l, v->r);
        if (v->l)
            v->l->rev ^= true;
        if (v->r)
            v->r->rev ^= true; 
    }
}

pair<pitem, pitem> split(pitem v, int k)
{
    if (!v)
        return make_pair(nullptr, nullptr);
    push(v);
    if (size_(v->l) >= k)
    {
        pair<pitem, pitem> p;
        p = split(v->l, k);
        v->l = p.second;
        update(v);
        return make_pair(p.first, v);
    }
    else
    {
        pair<pitem, pitem> p;
        p = split(v->r, k - size_(v->l) - 1);
        v->r = p.first;
        update(v);
        return make_pair(v, p.second);
    }
}

pitem merge(pitem l, pitem r)
{
    push(l);
    push(r);
    if (!l || !r)
    {
        return (l ? l : r);
    }
    if (l->y > r->y)
    {
        l->r = merge(l->r, r);
        update(l);
        return l;
    }
    else
    {
        r->l = merge(l, r->l);
        update(r);
        return r;
    }
}

void insert(int x, int ind)
{
    pitem t = new item(x);
    pair<pitem, pitem> p;
    p = split(root, ind);
    p.first = merge(p.first, t);
    //update(p.first);
    root = merge(p.first, p.second);
    //update(root);
}

int sum(int left, int right)
{
    pair<pitem, pitem> p;
    p = split(root, left);
    pair<pitem, pitem> pr;
    pr = split(p.second, right - left + 1);
    int res = sum_(pr.first);
    p.second = merge(pr.first, pr.second);
    //update(p.second);
    root = merge(p.first, p.second);
    //update(root);
    return res;
}

void rev(int left, int right)
{
    pair<pitem, pitem> p;
    p = split(root, left);
    pair<pitem, pitem> pr;
    pr = split(p.second, right - left + 1);
    if (pr.first)
    {
        pr.first->rev ^= true;
    }
    p.second = merge(pr.first, pr.second);
    //update(p.second);
    root = merge(p.first, p.second);
    //update(root);
}

void print_(pitem v)
{
    if (!v)
    {
        return;
    }
    cout << (v->x) << "|" << v->sum_ << " left: ";
    if (v->l)
    {
        cout << v->l->x << " ";
    }
    else
    {
        
        cout << "NULL ";
    }
    cout << "right ";
    if (v->r)
    {
        cout << v->r->x << " ";
    }
    else
    {
        
        cout << "NULL ";
    }
    cout << endl;
    print_(v->l);
    print_(v->r);
}

int main()
{
    srand(2);
    int n;
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        int x;
        cin >> x;
        insert(x, i);
    }
    int q;
    cin >> q;
    for (int j = 0; j < q; j++)
    {
        string c;
        cin >> c;
        if (c == "+")
        {
            int x, ind;
            cin >> x >> ind;
            insert(x, ind);
        }
        else if (c == "?")
        {
            int left, right;
            cin >> left >> right;
            cout << sum(left, right) << endl;
        }
        else if (c == "print")
        {
            print_(root);
 
        }
        else if (c == "rev")
        {
            int left, right;
            cin >> left >> right;
            rev(left, right);
        }
    }
    return 0;
}