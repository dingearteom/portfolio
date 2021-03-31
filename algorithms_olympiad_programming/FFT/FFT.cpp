#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

const double pi = acos(-1);

struct comp
{
    double a, b;
    comp(){};
    comp(double a, double b): a(a), b(b){};
};

comp operator * (comp x, comp y)
{
    return comp(x.a * y.a - x.b * y.b, x.a * y.b + x.b * y.a);
}

comp operator + (comp x, comp y)
{
    return comp(x.a + y.a, x.b + y.b);
}

comp operator -(comp x, comp y)
{
    return comp(x.a - y.a, x.b - y.b);
}

vector<comp> FFT(vector<comp> a, bool inv)
{
    if (a.size() == 1)
    {
        vector<comp> a_transformed;
        a_transformed.push_back(a[0]);
        return a_transformed;
    }

    int n = a.size() / 2;
    vector<comp> c;
    vector<comp> d;
    for (int i = 0; i < a.size(); i++)
    {
        if (i % 2 == 0)
        {
            c.push_back(a[i]);
        }
        else
        {
            d.push_back(a[i]);
        }       
    } 

    vector<comp> c_transformed = FFT(c, inv);
    vector<comp> d_tranformed = FFT(d, inv);
    vector<comp> a_transformed;
    a_transformed.resize(2 * n);
    double ang = ((2 * pi) / (2 * n)) * (inv ? -1 : 1);
    comp w = comp(cos(ang), sin(ang));
    comp t = comp(1, 0);
    for (int k = 0; k < n; k++)
    {
        a_transformed[k] = c_transformed[k] + t * d_tranformed[k];
        a_transformed[n + k] = c_transformed[k] - t * d_tranformed[k];
        t = t * w;              
    }
    return a_transformed;
}

string pro(string a, string b)
{
    bool sign = false;
    if (a[0] == '-')
    {
        sign ^= true;
        a = a.substr(1, a.size() -1);
    }
    if (b[0] == '-')
    {
        sign ^= true;
        b = b.substr(1, b.size() -1);
    }

    int s = a.size() + b.size();
    int t = 0;
    while ((1 << t) < s)
    {
        t++;
    }
    s = (1 << t);

    vector<comp> a_arr;
    vector<comp> b_arr;

    for (int i = a.size() - 1; i >= 0; i--)
    {
        a_arr.push_back(comp(a[i] - '0', 0));
    }
    for (int i = b.size() - 1; i >= 0; i--)
    {
        b_arr.push_back(comp(b[i] - '0', 0));
    }

    a_arr.resize(s, comp(0, 0));
    b_arr.resize(s, comp(0, 0));

    vector<comp> a_transformed = FFT(a_arr, false);
    vector<comp> b_transformed = FFT(b_arr, false);

    vector<comp> r_transformed;
    r_transformed.resize(s, comp(0, 0));
    for (int i = 0; i < s; i++)
    {
        r_transformed[i] = a_transformed[i] * b_transformed[i];
    }
    
    vector<comp> r = FFT(r_transformed, true);
    for (int i = 0; i < r.size(); i++)
    {
        r[i] = comp(r[i].a / s, r[i].b / s);
    }

    // processing before giving answer
    vector<int> res;
    res.resize(s, 0);
    for (int i = 0; i < r.size(); i++)
    {
        res[i] = round(r[i].a);
    }
    int ind = 0;
    while (ind < res.size())
    {
        if (res[ind] >= 10)
        {
            if (ind == res.size() - 1 )
            {
                res.push_back(res[ind] / 10);
            }
            else
            {
                res[ind + 1] += res[ind] / 10; 
            }
            res[ind] %= 10;
        }
        ind++;
    }
    ind = res.size() - 1;
    while (ind != 0 && res[ind] == 0)
    {
        res.pop_back();
        ind--;
    }

    // convert to string

    string ans = "";
    if (sign)
    {
        ans += '-';
    }
    for (int i = res.size() - 1; i >= 0; i--)
    {
        ans += char(res[i] + '0');
    } 
    return ans;
}

int main()
{
    string a, b;
    cin >> a;
    cin >> b;
    cout << pro(a, b);
    return 0;
}

