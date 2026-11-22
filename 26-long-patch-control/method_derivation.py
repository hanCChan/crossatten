"""Analytic architecture accounting; counts are not information guarantees."""
from s26_common import HERE,read,write,now


def main():
    n,channels,field,jump,center=8192,32,11,1,0.
    rows=[]
    for k,s,p,c in [(8,4,2,48),(8,4,2,64),(8,4,2,64),(4,4,0,64)]:
        center+=((k-1)/2-p)*jump
        field+=(k-1)*jump;jump*=s
        n=(n+2*p-k)//s+1
        rows.append(dict(kernel=k,stride=s,padding=p,input_channels=channels,output_channels=c,
            length=n,scalar_coordinates=n*c,local_support=field,token_jump=jump,center=center,
            conv_weights=k*channels*c,conv_biases=c,conv_macs=n*k*channels*c))
        channels=c
    shared=47082
    table=[]
    for arm,k,mid in [('P0',256,80),('LP',340,64)]:
        count=k*32*mid//8+mid+mid*64+64
        macs=32*(k*32*mid//8+mid*64)
        table.append(dict(arm=arm,params=shared+count,tokenizer_params=count,tokenizer_conv_macs=macs,
            scalar_coordinates_after_spatial_projection=32*mid,tokenizer_rank_upper_bound=min(mid,64)))
    token=sum(r['conv_weights']+r['conv_biases'] for r in rows)+2*(48+64+64)
    table.append(dict(arm='H0',params=shared+token,tokenizer_params=token,tokenizer_conv_macs=sum(r['conv_macs'] for r in rows)))
    assert token==86608 and table[-1]['params']==133690
    gate=read(HERE/'GATE.json')['architecture']
    shared_conv=8192*(7*5*32+5*32*32)
    for r in table:
        assert r['params']==gate[r['arm']]['params']
        r['shared_frontend_conv_macs']=shared_conv
        r['dispatcher_total_partial_macs']=gate[r['arm']]['partial_macs_per_packet']
        r['other_supported_partial_macs']=r['dispatcher_total_partial_macs']-shared_conv-r['tokenizer_conv_macs']
    assert field==350 and jump==256 and center==127.5 and n==32
    write(HERE/'METHOD_DERIVATION.json',dict(time=now(),status='COMPLETE_ANALYTIC_NO_FIT',
        recurrence=dict(length='floor((N+2p-d(k-1)-1)/s)+1',support='R_l=R_(l-1)+(k_l-1)d_l J_(l-1)',
            jump='J_l=s_l J_(l-1)',center='c_l=c_(l-1)+((k_l-1)d_l/2-p_l)J_(l-1)'),
        hierarchical_stages=rows,accounting=table,
        scope=['Local support describes the convolutional path with frozen BN; training BN uses batch/spatial statistics.',
               'TrueIB/FFT have global RF dependencies; this is not a physical isolated sample interval.',
               'Coordinate counts quantify representation dimensions, not mutual information or TX sufficiency.',
               'LP tokenizer is affine in its shared frontend input; HStem interleaves BN/GELU and convolutions. The complete networks are both nonlinear.',
               'Input-dependent Jacobians distinguish parameterizations but do not establish an RF-specific invariant or generalization theorem.',
               'Partial MACs exclude FFT, nonlinear and unsupported operations; lower parameters need not imply less compute.'],
        new_fits=0,new_waveform_reads=0))
    print('Analytic support, center, compression and parameter/MAC accounting agrees with audited models.')


if __name__=='__main__':main()
